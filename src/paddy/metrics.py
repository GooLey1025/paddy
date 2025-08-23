import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import metrics_utils

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


################################################################################
# Losses
################################################################################

class CombinedExpressionLoss(LossFunctionWrapper):
    """Combined loss function for tissue expression prediction.
    
    Combines MSE, correlation loss, and KL divergence to better capture
    relative expression patterns across tissues.
    
    Args:
        mse_weight: Weight for the MSE loss component.
        correlation_weight: Weight for the correlation loss component.
        kl_weight: Weight for the KL divergence loss component.
        contrastive_weight: Weight for the contrastive loss component.
        contrastive_temp: Temperature parameter for contrastive loss.
        epsilon: Small value to avoid numerical issues.
        scale_gradients: Whether to scale gradients for stability.
        simplified: Whether to use a simplified version of the loss.
    """
    
    def __init__(
        self,
        mse_weight=0.7,
        correlation_weight=0.2,
        kl_weight=0.1,
        contrastive_weight=0.0,  # Disabled by default as it can be unstable
        contrastive_temp=0.1,
        epsilon=1e-6,
        scale_gradients=True,
        simplified=False,
        reduction=losses_utils.ReductionV2.AUTO,
        name="combined_expression_loss",
    ):
        self.mse_weight = mse_weight
        self.correlation_weight = correlation_weight
        self.kl_weight = kl_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temp = contrastive_temp
        self.epsilon = epsilon
        self.scale_gradients = scale_gradients
        self.simplified = simplified
        
        combined_loss = lambda y_true, y_pred: self._combined_loss(y_true, y_pred)
        super(CombinedExpressionLoss, self).__init__(combined_loss,
                                                    name=name,
                                                    reduction=reduction)
    
    def _combined_loss(self, y_true, y_pred):
        # Apply gradient scaling for stability if requested
        if self.scale_gradients:
            y_pred = tf.stop_gradient(y_pred) + 0.1 * (y_pred - tf.stop_gradient(y_pred))
        
        # Ensure positive values for stability
        y_true = tf.maximum(y_true, self.epsilon)
        y_pred = tf.maximum(y_pred, self.epsilon)
        
        # MSE component
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # Use simplified version if requested (only MSE + correlation)
        if self.simplified:
            # Correlation component (1 - Pearson correlation)
            y_true_centered = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
            y_pred_centered = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)
            
            y_true_std = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered), axis=-1) + self.epsilon)
            y_pred_std = tf.sqrt(tf.reduce_sum(tf.square(y_pred_centered), axis=-1) + self.epsilon)
            
            correlation = tf.reduce_sum(y_true_centered * y_pred_centered, axis=-1) / (y_true_std * y_pred_std + self.epsilon)
            correlation = tf.clip_by_value(correlation, -1.0 + self.epsilon, 1.0 - self.epsilon)
            correlation_loss = 1.0 - correlation
            
            # Combine only MSE and correlation for stability
            return self.mse_weight * mse_loss + self.correlation_weight * correlation_loss
        
        # Full version with all components
        # Correlation component (1 - Pearson correlation)
        y_true_centered = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
        y_pred_centered = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        
        y_true_std = tf.sqrt(tf.reduce_sum(tf.square(y_true_centered), axis=-1) + self.epsilon)
        y_pred_std = tf.sqrt(tf.reduce_sum(tf.square(y_pred_centered), axis=-1) + self.epsilon)
        
        correlation = tf.reduce_sum(y_true_centered * y_pred_centered, axis=-1) / (y_true_std * y_pred_std + self.epsilon)
        correlation = tf.clip_by_value(correlation, -1.0 + self.epsilon, 1.0 - self.epsilon)
        correlation_loss = 1.0 - correlation
        
        # KL divergence component
        # Add epsilon and normalize to sum to 1
        y_true_norm = y_true + self.epsilon
        y_pred_norm = y_pred + self.epsilon
        
        y_true_norm = y_true_norm / tf.reduce_sum(y_true_norm, axis=-1, keepdims=True)
        y_pred_norm = y_pred_norm / tf.reduce_sum(y_pred_norm, axis=-1, keepdims=True)
        
        kl_loss = tf.reduce_sum(y_true_norm * tf.math.log(y_true_norm / y_pred_norm + self.epsilon), axis=-1)
        kl_loss = tf.clip_by_value(kl_loss, 0.0, 10.0)  # Limit extreme values
        
        # Contrastive loss component only if weight > 0
        contrastive_loss = 0.0
        if self.contrastive_weight > 0:
            # Normalize vectors for cosine similarity
            y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
            y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
            
            # Calculate similarity matrix
            batch_size = tf.shape(y_true)[0]
            similarity_matrix = tf.matmul(y_pred_norm, tf.transpose(y_true_norm))
            
            # Apply temperature scaling
            similarity_matrix /= self.contrastive_temp
            
            # Create labels - diagonal elements are positive pairs
            labels = tf.eye(batch_size)
            
            # Calculate contrastive loss (similar to NT-Xent loss)
            contrastive_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=similarity_matrix)
            )
            contrastive_loss = tf.clip_by_value(contrastive_loss, 0.0, 10.0)  # Limit extreme values
        
        # Combine all losses
        combined = (
            self.mse_weight * mse_loss + 
            self.correlation_weight * correlation_loss + 
            self.kl_weight * kl_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        return combined


class TissueRelationshipLoss(LossFunctionWrapper):
    """Loss function that focuses on tissue expression relationships.
    
    This loss encourages the model to learn the relative expression patterns
    between different tissues, which is important for capturing tissue-specific
    expression profiles.
    
    Args:
        tissue_graph: Optional adjacency matrix representing known tissue relationships.
        alpha: Weight for the tissue relationship component.
        epsilon: Small value to avoid numerical issues.
        scale_gradients: Whether to scale gradients for stability.
        mse_weight: Weight for the base MSE component.
        relationship_weight: Weight for the tissue relationship component.
        rank_weight: Weight for the rank correlation component.
        use_rank_correlation: Whether to include rank correlation in the loss.
    """
    
    def __init__(
        self,
        tissue_graph=None,
        alpha=0.5,
        epsilon=1e-6,
        scale_gradients=True,
        mse_weight=0.6,
        relationship_weight=0.3,
        rank_weight=0.1,
        use_rank_correlation=True,
        reduction=losses_utils.ReductionV2.AUTO,
        name="tissue_relationship_loss",
    ):
        self.tissue_graph = tissue_graph
        self.alpha = alpha
        self.epsilon = epsilon
        self.scale_gradients = scale_gradients
        self.mse_weight = mse_weight
        self.relationship_weight = relationship_weight
        self.rank_weight = rank_weight
        self.use_rank_correlation = use_rank_correlation
        
        tissue_loss = lambda y_true, y_pred: self._tissue_loss(y_true, y_pred)
        super(TissueRelationshipLoss, self).__init__(tissue_loss,
                                                    name=name,
                                                    reduction=reduction)
    
    def _tissue_loss(self, y_true, y_pred):
        # Apply gradient scaling for stability if requested
        if self.scale_gradients:
            y_pred = tf.stop_gradient(y_pred) + 0.1 * (y_pred - tf.stop_gradient(y_pred))
        
        # Ensure positive values for stability
        y_true = tf.maximum(y_true, self.epsilon)
        y_pred = tf.maximum(y_pred, self.epsilon)
        
        # Base MSE loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # Calculate tissue relationship matrix for true values
        # Higher values indicate more similar expression patterns
        num_tissues = tf.shape(y_true)[-1]
        
        # Create tissue relationship matrices
        # For each sample, compute pairwise differences between tissues
        y_true_expanded_1 = tf.expand_dims(y_true, axis=-1)  # [batch, tissues, 1]
        y_true_expanded_2 = tf.expand_dims(y_true, axis=-2)  # [batch, 1, tissues]
        
        # Calculate true tissue relationships (differences)
        true_tissue_rel = tf.abs(y_true_expanded_1 - y_true_expanded_2)  # [batch, tissues, tissues]
        
        # Same for predictions
        y_pred_expanded_1 = tf.expand_dims(y_pred, axis=-1)
        y_pred_expanded_2 = tf.expand_dims(y_pred, axis=-2)
        pred_tissue_rel = tf.abs(y_pred_expanded_1 - y_pred_expanded_2)
        
        # Calculate the loss on tissue relationships
        relationship_loss = tf.reduce_mean(tf.square(true_tissue_rel - pred_tissue_rel))
        
        # If we have a tissue graph, use it to weight the relationships
        if self.tissue_graph is not None:
            # tissue_graph should be a tensor of shape [num_tissues, num_tissues]
            # with values indicating the strength of relationship
            weighted_rel_loss = tf.reduce_mean(
                tf.multiply(tf.square(true_tissue_rel - pred_tissue_rel), self.tissue_graph)
            )
            relationship_loss = weighted_rel_loss
        
        # Add rank correlation component if requested
        rank_loss = 0.0
        if self.use_rank_correlation:
            # Calculate rank correlations between tissues
            # This helps preserve the ordering of expression levels across tissues
            
            # Create rank matrices (approximate using softmax-based sorting)
            def soft_rank(x):
                # Softmax-based differentiable ranking
                # Higher values get higher ranks
                x_softmax = tf.nn.softmax(x * 10, axis=-1)  # Scale for sharper softmax
                return tf.cumsum(x_softmax, axis=-1)
            
            # Get approximate ranks
            y_true_ranks = soft_rank(y_true)
            y_pred_ranks = soft_rank(y_pred)
            
            # Calculate rank correlation loss (simplified Spearman)
            rank_loss = tf.reduce_mean(tf.square(y_true_ranks - y_pred_ranks))
        
        # Combine losses with weights
        combined_loss = (
            self.mse_weight * mse_loss + 
            self.relationship_weight * relationship_loss + 
            self.rank_weight * rank_loss
        )
        
        return combined_loss


def mean_squared_error_udot(y_true, y_pred, udot_weight: float = 1):
    """Mean squared error with mean-normalized specificity term.

    Args:
        udot_weight: Weight of the mean-normalized specificity term.
    """
    mse_term = tf.keras.losses.mean_squared_error(y_true, y_pred)

    yn_true = y_true - tf.math.reduce_mean(y_true, axis=-1, keepdims=True)
    yn_pred = y_pred - tf.math.reduce_mean(y_pred, axis=-1, keepdims=True)
    udot_term = -tf.reduce_mean(yn_true * yn_pred, axis=-1)

    return mse_term + udot_weight * udot_term


class MeanSquaredErrorUDot(LossFunctionWrapper):
    """Mean squared error with mean-normalized specificity term.

    Args:
        udot_weight: Weight of the mean-normalized specificity term.
    """

    def __init__(
        self,
        udot_weight: float = 1,
        reduction=losses_utils.ReductionV2.AUTO,
        name: str = "mse_udot",
    ):
        self.udot_weight = udot_weight
        mse_udot = lambda yt, yp: mean_squared_error_udot(
            yt, yp, self.udot_weight)
        super(MeanSquaredErrorUDot, self).__init__(mse_udot,
                                                   name=name,
                                                   reduction=reduction)


def poisson_kl(y_true, y_pred, kl_weight=1, epsilon=1e-7):
    """Poisson decomposition with KL specificity term.

    Args:
        kl_weight (float): Weight of the KL specificity term.
        epsilon (float): Added small value to avoid log(0).
    """
    # poisson loss
    poisson_term = tf.keras.losses.poisson(y_true, y_pred)

    # add epsilon to protect against all tiny values
    y_true += epsilon
    y_pred += epsilon

    # normalize to sum to one
    yn_true = y_true / tf.math.reduce_sum(y_true, axis=-1, keepdims=True)
    yn_pred = y_pred / tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)

    # kl term
    kl_term = tf.keras.losses.kl_divergence(yn_true, yn_pred)

    # weighted combination
    return poisson_term + kl_weight * kl_term


class PoissonKL(LossFunctionWrapper):
    """Possion decomposition with KL specificity term.

    Args:
      kl_weight (float): Weight of the KL specificity term.
    """

    def __init__(
        self,
        kl_weight: int = 1,
        reduction=losses_utils.ReductionV2.AUTO,
        name="poisson_kl",
    ):
        self.kl_weight = kl_weight
        pois_kl = lambda yt, yp: poisson_kl(yt, yp, self.kl_weight)
        super(PoissonKL, self).__init__(pois_kl,
                                        name=name,
                                        reduction=reduction)


def poisson(yt, yp, epsilon: float = 1e-7):
    """Poisson loss, without mean reduction."""
    return yp - yt * tf.math.log(yp + epsilon)


def poisson_multinomial(
    y_true,
    y_pred,
    total_weight: float = 1,
    weight_range: float = 1,
    weight_exp: int = 4,
    epsilon: float = 1e-7,
    rescale: bool = False,
):
    """Possion decomposition with multinomial specificity term.

    Args:
        total_weight (float): Weight of the Poisson total term.
        epsilon (float): Added small value to avoid log(0).
        rescale (bool): Rescale loss after re-weighting.
    """
    if len(y_true.shape) == 1:
        raise ValueError("poisson_multinomial is not suitable for predicting 1d output, change to mse loss")
    seq_len = y_true.shape[1]

    if weight_range < 1:
        raise ValueError("Poisson Multinomial weight_range must be >=1")
    elif weight_range == 1:
        position_weights = tf.ones((1, seq_len, 1))
    else:
        pos_start = -(seq_len / 2 - 0.5)
        pos_end = seq_len / 2 + 0.5
        positions = tf.range(pos_start, pos_end, dtype=tf.float32)
        sigma = -pos_start / (np.log(weight_range))**(1 / weight_exp)
        position_weights = tf.exp(-((positions / sigma)**weight_exp))
        position_weights /= tf.reduce_max(position_weights)
        position_weights = tf.expand_dims(position_weights, axis=0)
        position_weights = tf.expand_dims(position_weights, axis=-1)

    y_true = tf.math.multiply(y_true, position_weights)
    y_pred = tf.math.multiply(y_pred, position_weights)

    # sum across lengths
    s_true = tf.math.reduce_sum(y_true, axis=-2)  # B x T
    s_pred = tf.math.reduce_sum(y_pred, axis=-2)  # B x T

    # total count poisson loss, mean across targets
    poisson_term = poisson(s_true, s_pred)  # B x T
    poisson_term /= tf.reduce_sum(position_weights)

    # add epsilon to protect against tiny values
    y_true += epsilon
    y_pred += epsilon

    # normalize to sum to one
    p_pred = y_pred / tf.expand_dims(s_pred, axis=-2)  # B x L x T

    # multinomial loss
    pl_pred = tf.math.log(p_pred)  # B x L x T
    multinomial_dot = -tf.math.multiply(y_true, pl_pred)  # B x L x T
    multinomial_term = tf.math.reduce_sum(multinomial_dot, axis=-2)  # B x T
    multinomial_term /= tf.reduce_sum(position_weights)

    # normalize to scale of 1:1 term ratio
    loss_raw = multinomial_term + total_weight * poisson_term  # B x T
    if rescale:
        loss_rescale = loss_raw * 2 / (1 + total_weight)
    else:
        loss_rescale = loss_raw

    return loss_rescale


class PoissonMultinomial(LossFunctionWrapper):
    """Possion decomposition with multinomial specificity term.

    Args:
      total_weight (float): Weight of the Poisson total term.
    """

    def __init__(
        self,
        total_weight: float = 1,
        weight_range: float = 1,
        weight_exp: int = 4,
        reduction=losses_utils.ReductionV2.AUTO,
        name: str = "poisson_multinomial",
    ):
        pois_mn = lambda yt, yp: poisson_multinomial(yt, yp, total_weight,
                                                     weight_range, weight_exp)
        super(PoissonMultinomial, self).__init__(pois_mn,
                                                 name=name,
                                                 reduction=reduction)

################################################################################
# Metrics
################################################################################
class SeqAUC(tf.keras.metrics.AUC):
    """AUC metric for multi-task sequence data.

    Args:
      curve (str): Metric type--'ROC' or 'PR'.
      summarize (bool): Whether to summarize over all tasks.
    """

    def __init__(self,
                 curve: str = "ROC",
                 name: str = None,
                 summarize: bool = True,
                 **kwargs):
        if name is None:
            if curve == "ROC":
                name = "auroc"
            elif curve == "PR":
                name = "auprc"
        super(SeqAUC, self).__init__(curve=curve,
                                     name=name,
                                     multi_label=True,
                                     **kwargs)
        self._summarize = summarize

    def update_state(self, y_true, y_pred, **kwargs):
        """Flatten sequence length before update."""

        # flatten batch and sequence length
        num_targets = y_pred.shape[-1]
        y_true = tf.reshape(y_true, (-1, num_targets))
        y_pred = tf.reshape(y_pred, (-1, num_targets))

        # update
        super(SeqAUC, self).update_state(y_true, y_pred, **kwargs)

    def interpolate_pr_auc(self):
        """Add option to remove summary."""
        dtp = self.true_positives[:self.num_thresholds -
                                  1] - self.true_positives[1:]
        p = tf.math.add(self.true_positives, self.false_positives)
        dp = p[:self.num_thresholds - 1] - p[1:]
        prec_slope = tf.math.divide_no_nan(dtp,
                                           tf.maximum(dp, 0),
                                           name="prec_slope")
        intercept = self.true_positives[1:] - tf.multiply(prec_slope, p[1:])

        safe_p_ratio = tf.where(
            tf.logical_and(p[:self.num_thresholds - 1] > 0, p[1:] > 0),
            tf.math.divide_no_nan(
                p[:self.num_thresholds - 1],
                tf.maximum(p[1:], 0),
                name="recall_relative_ratio",
            ),
            tf.ones_like(p[1:]),
        )

        pr_auc_increment = tf.math.divide_no_nan(
            prec_slope * (dtp + intercept * tf.math.log(safe_p_ratio)),
            tf.maximum(self.true_positives[1:] + self.false_negatives[1:], 0),
            name="pr_auc_increment",
        )

        if self.multi_label:
            by_label_auc = tf.reduce_sum(pr_auc_increment,
                                         name=self.name + "_by_label",
                                         axis=0)

            if self._summarize:
                if self.label_weights is None:
                    # Evenly weighted average of the label AUCs.
                    return tf.reduce_mean(by_label_auc, name=self.name)
                else:
                    # Weighted average of the label AUCs.
                    return tf.math.divide_no_nan(
                        tf.reduce_sum(
                            tf.multiply(by_label_auc, self.label_weights)),
                        tf.reduce_sum(self.label_weights),
                        name=self.name,
                    )
            else:
                return by_label_auc
        else:
            if self._summarize:
                return tf.reduce_sum(pr_auc_increment,
                                     name="interpolate_pr_auc")
            else:
                return pr_auc_increment

    def result(self):
        """Add option to remove summary.
        It's not clear why, but these metrics_utils == aren't working for tf2.6 on.
        I'm hacking a solution to compare the values instead."""
        if (self.curve.value == metrics_utils.AUCCurve.PR.value
                and self.summation_method.value
                == metrics_utils.AUCSummationMethod.INTERPOLATION.value):
            # This use case is different and is handled separately.
            return self.interpolate_pr_auc()

        # Set `x` and `y` values for the curves based on `curve` config.
        recall = tf.math.divide_no_nan(
            self.true_positives,
            tf.math.add(self.true_positives, self.false_negatives))
        if self.curve.value == metrics_utils.AUCCurve.ROC.value:
            fp_rate = tf.math.divide_no_nan(
                self.false_positives,
                tf.math.add(self.false_positives, self.true_negatives),
            )
            x = fp_rate
            y = recall
        else:  # curve == 'PR'.
            precision = tf.math.divide_no_nan(
                self.true_positives,
                tf.math.add(self.true_positives, self.false_positives),
            )
            x = recall
            y = precision

        # Find the rectangle heights based on `summation_method`.
        if (self.summation_method.value ==
                metrics_utils.AUCSummationMethod.INTERPOLATION.value):
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.0
        elif (self.summation_method.value ==
              metrics_utils.AUCSummationMethod.MINORING.value):
            heights = tf.minimum(y[:self.num_thresholds - 1], y[1:])
        else:  # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
            heights = tf.maximum(y[:self.num_thresholds - 1], y[1:])

        # Sum up the areas of all the rectangles.
        if self.multi_label:
            riemann_terms = tf.multiply(x[:self.num_thresholds - 1] - x[1:],
                                        heights)
            by_label_auc = tf.reduce_sum(riemann_terms,
                                         name=self.name + "_by_label",
                                         axis=0)

            if self._summarize:
                if self.label_weights is None:
                    # Unweighted average of the label AUCs.
                    return tf.reduce_mean(by_label_auc, name=self.name)
                else:
                    # Weighted average of the label AUCs.
                    return tf.math.div_no_nan(
                        tf.reduce_sum(
                            tf.multiply(by_label_auc, self.label_weights)),
                        tf.reduce_sum(self.label_weights),
                        name=self.name,
                    )
            else:
                return by_label_auc
        else:
            if self._summarize:
                return tf.reduce_sum(
                    tf.multiply(x[:self.num_thresholds - 1] - x[1:], heights),
                    name=self.name,
                )
            else:
                return tf.multiply(x[:self.num_thresholds - 1] - x[1:],
                                   heights)


class PearsonR(tf.keras.metrics.Metric):
    """PearsonR metric for multi-task data.

    Args:
      num_targets (int): Number of tasks.
      summarize (bool): Whether to summarize over all tasks.
    """

    def __init__(self, num_targets, summarize=True, name="pearsonr", **kwargs):
        super(PearsonR, self).__init__(name=name, **kwargs)
        self._summarize = summarize
        if num_targets == 1:
            self._shape = () # scalar output
        else:
            self._shape = (num_targets, )
        self._count = self.add_weight(name="count",
                                      shape=self._shape,
                                      initializer="zeros")

        self._product = self.add_weight(name="product",
                                        shape=self._shape,
                                        initializer="zeros")
        self._true_sum = self.add_weight(name="true_sum",
                                         shape=self._shape,
                                         initializer="zeros")
        self._true_sumsq = self.add_weight(name="true_sumsq",
                                           shape=self._shape,
                                           initializer="zeros")
        self._pred_sum = self.add_weight(name="pred_sum",
                                         shape=self._shape,
                                         initializer="zeros")
        self._pred_sumsq = self.add_weight(name="pred_sumsq",
                                           shape=self._shape,
                                           initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state for a batch."""
        y_true = tf.cast(y_true, "float32")
        y_pred = tf.cast(y_pred, "float32")

        if len(y_true.shape) == 2:
            reduce_axes = 0
        elif len(y_true.shape) == 1:
            reduce_axes = 0
        else:
            reduce_axes = [0, 1]

        product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=reduce_axes)
        self._product.assign_add(product)

        true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
        self._true_sum.assign_add(true_sum)

        true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=reduce_axes)
        self._true_sumsq.assign_add(true_sumsq)

        pred_sum = tf.reduce_sum(y_pred, axis=reduce_axes)
        self._pred_sum.assign_add(pred_sum)

        pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=reduce_axes)
        self._pred_sumsq.assign_add(pred_sumsq)

        count = tf.ones_like(y_true)
        count = tf.reduce_sum(count, axis=reduce_axes)
        self._count.assign_add(count)

    def result(self):
        """Compute PearsonR result from state."""
        true_mean = tf.divide(self._true_sum, self._count)
        true_mean2 = tf.math.square(true_mean)
        pred_mean = tf.divide(self._pred_sum, self._count)
        pred_mean2 = tf.math.square(pred_mean)

        term1 = self._product
        term2 = -tf.multiply(true_mean, self._pred_sum)
        term3 = -tf.multiply(pred_mean, self._true_sum)
        term4 = tf.multiply(self._count, tf.multiply(true_mean, pred_mean))
        covariance = term1 + term2 + term3 + term4

        true_var = self._true_sumsq - tf.multiply(self._count, true_mean2)
        pred_var = self._pred_sumsq - tf.multiply(self._count, pred_mean2)
        pred_var = tf.where(tf.greater(pred_var, 1e-12), pred_var,
                            np.inf * tf.ones_like(pred_var))

        tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
        correlation = tf.divide(covariance, tp_var)

        if self._summarize:
            return tf.reduce_mean(correlation)
        else:
            return correlation

    def reset_state(self):
        """Reset metric state."""
        K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])


class R2(tf.keras.metrics.Metric):
    """R2 metric for multi-task data.

    Args:
      num_targets (int): Number of tasks.
      summarize (bool): Whether to summarize over all tasks.
    """

    def __init__(self, num_targets, summarize=True, name="r2", **kwargs):
        super(R2, self).__init__(name=name, **kwargs)
        self._summarize = summarize
        if num_targets == 1:
            self._shape = () # scalar output
        else:
            self._shape = (num_targets, )
        self._count = self.add_weight(name="count",
                                      shape=self._shape,
                                      initializer="zeros")

        self._true_sum = self.add_weight(name="true_sum",
                                         shape=self._shape,
                                         initializer="zeros")
        self._true_sumsq = self.add_weight(name="true_sumsq",
                                           shape=self._shape,
                                           initializer="zeros")

        self._product = self.add_weight(name="product",
                                        shape=self._shape,
                                        initializer="zeros")
        self._pred_sumsq = self.add_weight(name="pred_sumsq",
                                           shape=self._shape,
                                           initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state for a batch."""
        y_true = tf.cast(y_true, "float32")
        y_pred = tf.cast(y_pred, "float32")

        if len(y_true.shape) == 1:
            reduce_axes = 0
        elif len(y_true.shape) == 2:
            reduce_axes = 0
        else:
            reduce_axes = [0, 1]

        true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
        self._true_sum.assign_add(true_sum)

        true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=reduce_axes)
        self._true_sumsq.assign_add(true_sumsq)

        product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=reduce_axes)
        self._product.assign_add(product)

        pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=reduce_axes)
        self._pred_sumsq.assign_add(pred_sumsq)

        count = tf.ones_like(y_true)
        count = tf.reduce_sum(count, axis=reduce_axes)
        self._count.assign_add(count)
        
    def result(self):
        """Compute R2 result from state."""
        true_mean = tf.divide(self._true_sum, self._count)
        true_mean2 = tf.math.square(true_mean)

        total = self._true_sumsq - tf.multiply(self._count, true_mean2)
        total = tf.where(tf.greater(total, 1e-12), total, np.inf * tf.ones_like(total))

        resid1 = self._pred_sumsq
        resid2 = -2 * self._product
        resid3 = self._true_sumsq
        resid = resid1 + resid2 + resid3
        resid = tf.maximum(resid, 0.0) # add

        r2 = 1.0 - tf.divide(resid, total)

        if self._summarize:
            return tf.reduce_mean(r2)
        else:
            return r2

    def reset_state(self):
        """Reset metric state."""
        K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])


class MSEPlusPearsonLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.1, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.mse = tf.keras.losses.MeanSquaredError(reduction=reduction)

    def call(self, y_true, y_pred):
        mse_loss = self.mse(y_true, y_pred)

        x = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        y = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
        r_num = tf.reduce_sum(x * y, axis=-1)
        r_den = tf.sqrt(tf.reduce_sum(x ** 2, axis=-1)) * tf.sqrt(tf.reduce_sum(y ** 2, axis=-1))
        r = r_num / (r_den + 1e-6)
        pearson_loss = 1 - r

        return mse_loss + self.alpha * pearson_loss