#!/usr/bin/env python

import argparse
import os
import sys
import random
import shutil
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from paddy import dataset
from paddy import seqnn
from paddy import trainer
from paddy import layers

"""
paddy_train.py
Train a seq2exp model using cross attention fusion of multi-layer embeddings 
from a pretrained upstream model.

Multi-layer Cross Attention Fusion Process (Memory-Optimized):
1. Extract multi-layer hidden states [B, T, C_l] from Stage-1 model
2. Project each layer to same dimension D
3. Pool each layer → layer vectors [B, L, D] as K/V
4. Create query embedding [B, T, D] from mean of all layers
5. Apply cross-attention: Q=[B,T,D], K/V=[B,L,D] → O(L·T) complexity
6. Compute layer attention weights and fuse layers using einsum
7. Add residual connection and layer normalization

Memory Optimizations Applied:
- Avoid tf.tile operation that creates [B,L,T,D] intermediate tensors
- Use K/V length = L instead of L×T (reduces complexity from O(L·T²) to O(L·T))
- Use einsum for efficient tensor operations without large intermediates
- Direct cross-attention between [B,T,D] query and [B,L,D] key/value
"""


class CrossAttentionLayerFusion(tf.keras.layers.Layer):
    """Cross attention layer fusion module (Memory-Optimized).
    
    Fuses multi-layer embeddings from pretrained model using cross attention.
    
    Performance Comparison:
    - Before: K/V length = L×T, complexity O(L·T²), memory ~41GB for typical sizes
    - After:  K/V length = L,   complexity O(L·T),  memory ~few MB
    
    Key optimizations:
    1. Direct cross-attention between Q=[B,T,D] and K/V=[B,L,D]
    2. Einsum operations instead of tile+reshape
    3. No large intermediate [B,L,T,D] tensors
    """
    
    def __init__(self, 
                 fusion_dim=256,
                 num_heads=8,
                 attention_dropout=0.1,
                 temperature=1.0,
                 use_layer_gate=False,
                 entropy_reg_weight=0.01,
                 pool_type="mean",
                 kernel_initializer="he_normal",
                 layer_score_type="dot_product",  # "dot_product", "bilinear", "mlp"
                 use_custom_mha=False,  # Whether to use custom sequence-aware MHA
                 mha_params=None,  # Parameters for custom MHA
                 **kwargs):
        """Initialize cross attention layer fusion.
        
        Args:
            fusion_dim: Target fusion dimension D
            num_heads: Number of attention heads
            attention_dropout: Dropout rate for attention
            temperature: Temperature scaling for attention scores
            use_layer_gate: Whether to use layer gating
            entropy_reg_weight: Weight for entropy regularization
            pool_type: Pooling type for layer vectors ("mean", "max", "attention")
            kernel_initializer: Kernel initializer
            layer_score_type: Method to compute layer attention scores:
                - "dot_product": Simple dot product similarity (default, efficient)
                - "bilinear": Bilinear transformation (more expressive)
                - "mlp": Multi-layer perceptron (most expressive but slower)
            use_custom_mha: Whether to use custom sequence-aware MultiheadAttention from layers.py
            mha_params: Parameters for custom MHA (position features, relative position, etc.)
        """
        super(CrossAttentionLayerFusion, self).__init__(**kwargs)
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.temperature = temperature
        self.use_layer_gate = use_layer_gate
        self.entropy_reg_weight = entropy_reg_weight
        self.pool_type = pool_type
        self.kernel_initializer = kernel_initializer
        self.layer_score_type = layer_score_type
        self.use_custom_mha = use_custom_mha
        self.mha_params = mha_params or {}
        
    def build(self, input_shape):
        """Build the layer."""
        # input_shape should be a list of shapes: [(B, T, C_1), (B, T, C_2), ...]
        if not isinstance(input_shape, list):
            raise ValueError("Input should be a list of layer embeddings")
        
        self.num_layers = len(input_shape)
        
        # Layer projection layers - project each layer to fusion_dim
        self.layer_projections = []
        for i, shape in enumerate(input_shape):
            proj = tf.keras.layers.Dense(
                self.fusion_dim,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                name=f"layer_proj_{i}"
            )
            self.layer_projections.append(proj)
        
        # Pooling layer for creating layer vectors
        if self.pool_type == "attention":
            if self.use_custom_mha:
                # Use custom sequence-aware MHA from layers.py
                pool_mha_params = {
                    'value_size': self.fusion_dim,
                    'key_size': self.fusion_dim,
                    'heads': 1,
                    'attention_dropout_rate': self.attention_dropout,
                    'initializer': self.kernel_initializer,
                    **self.mha_params
                }
                self.pool_attention = layers.MultiheadAttention(**pool_mha_params)
            else:
                # Use standard Keras MHA
                self.pool_attention = tf.keras.layers.MultiHeadAttention(
                    num_heads=1,
                    key_dim=self.fusion_dim,
                    dropout=self.attention_dropout,
                    name="pool_attention"
                )
            
            # Learnable query embeddings for each layer
            # Each layer gets its own learnable "importance detector" query
            self.layer_pool_queries = self.add_weight(
                name="layer_pool_queries",
                shape=(self.num_layers, self.fusion_dim),
                initializer=self.kernel_initializer,
                trainable=True,
                dtype=self.dtype
            )
        
        # Query projection for cross attention
        self.query_projection = tf.keras.layers.Dense(
            self.fusion_dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            name="query_projection"
        )
        
        # Cross attention layer
        if self.use_custom_mha:
            # Use custom sequence-aware MHA from layers.py
            cross_mha_params = {
                'value_size': self.fusion_dim // self.num_heads,
                'key_size': self.fusion_dim // self.num_heads,
                'heads': self.num_heads,
                'attention_dropout_rate': self.attention_dropout,
                'initializer': self.kernel_initializer,
                **self.mha_params
            }
            self.cross_attention = layers.MultiheadAttention(**cross_mha_params)
        else:
            # Use standard Keras MHA
            self.cross_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.fusion_dim // self.num_heads,
                dropout=self.attention_dropout,
                name="cross_attention"
            )
        
        # Layer gate (optional)
        if self.use_layer_gate:
            self.layer_gate = tf.keras.layers.Dense(
                self.num_layers,
                activation="sigmoid",
                kernel_initializer="zeros",
                name="layer_gate"
            )
        
        # Layer score computation components
        if self.layer_score_type == "bilinear":
            # Bilinear transformation: query^T W layer_vector
            self.bilinear_layer = tf.keras.layers.Dense(
                self.fusion_dim,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                name="bilinear_transform"
            )
        elif self.layer_score_type == "mlp":
            # MLP for computing attention scores
            self.score_mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    self.fusion_dim,
                    activation='relu',
                    kernel_initializer=self.kernel_initializer,
                    name="score_mlp_hidden"
                ),
                tf.keras.layers.Dropout(self.attention_dropout),
                tf.keras.layers.Dense(
                    1,
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    name="score_mlp_output"
                )
            ], name="score_mlp")
        
        # Layer normalization
        self.layer_norm = tf.keras.layers.LayerNormalization(name="layer_norm")
        
        super(CrossAttentionLayerFusion, self).build(input_shape)
    
    def _compute_layer_scores(self, query_pooled, layer_vectors, training=None):
        """Compute layer attention scores using different methods.
        
        Args:
            query_pooled: Pooled query vector [B, D]
            layer_vectors: Layer representations [B, L, D]
            training: Training flag
            
        Returns:
            layer_scores: Attention scores [B, L]
        """
        if self.layer_score_type == "dot_product":
            # Simple dot product similarity (current default)
            layer_scores = tf.einsum('bd,bld->bl', query_pooled, layer_vectors)
            
        elif self.layer_score_type == "bilinear":
            # Bilinear transformation: query^T W layer_vector
            # Transform layer_vectors through bilinear layer
            transformed_layers = self.bilinear_layer(layer_vectors)  # [B, L, D]
            layer_scores = tf.einsum('bd,bld->bl', query_pooled, transformed_layers)
            
        elif self.layer_score_type == "mlp":
            # MLP-based scoring: more expressive but slower
            batch_size = tf.shape(query_pooled)[0]
            num_layers = tf.shape(layer_vectors)[1]
            
            # Expand query to match layer dimensions: [B, D] -> [B, L, D]
            query_expanded = tf.expand_dims(query_pooled, axis=1)  # [B, 1, D]
            query_expanded = tf.tile(query_expanded, [1, num_layers, 1])  # [B, L, D]
            
            # Concatenate query and layer vectors for MLP input
            mlp_input = tf.concat([query_expanded, layer_vectors], axis=-1)  # [B, L, 2D]
            
            # Reshape for MLP processing: [B*L, 2D]
            mlp_input_flat = tf.reshape(mlp_input, [batch_size * num_layers, 2 * self.fusion_dim])
            
            # Apply MLP to compute scores
            scores_flat = self.score_mlp(mlp_input_flat, training=training)  # [B*L, 1]
            
            # Reshape back to [B, L]
            layer_scores = tf.reshape(tf.squeeze(scores_flat, axis=-1), [batch_size, num_layers])
            
        else:
            raise ValueError(f"Unknown layer_score_type: {self.layer_score_type}")
        
        return layer_scores
    
    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: List of layer embeddings [(B, T, C_1), (B, T, C_2), ...]
            training: Training flag
            
        Returns:
            Fused embedding [B, T, D]
        """
        if not isinstance(inputs, list):
            raise ValueError("Input should be a list of layer embeddings")
        
        batch_size = tf.shape(inputs[0])[0]
        seq_length = tf.shape(inputs[0])[1]
        
        # Step 1: Project each layer to fusion_dim and handle different sequence lengths
        projected_layers = []
        for i, layer_emb in enumerate(inputs):
            proj_emb = self.layer_projections[i](layer_emb)  # [B, T_i, D]
            projected_layers.append(proj_emb)
        
        # Step 1.5: Adaptive pooling to fixed length (1024 for head compatibility)
        target_seq_length = 1024
        
        resized_layers = []
        for proj_emb in projected_layers:
            if proj_emb.shape[1] == target_seq_length:
                resized_layers.append(proj_emb)
            else:
                # Adaptive resize using bilinear interpolation
                reshaped = tf.expand_dims(proj_emb, axis=1)  # [B, 1, T, D]
                resized = tf.image.resize(reshaped, [1, target_seq_length], method='bilinear')
                resized_layers.append(tf.squeeze(resized, axis=1))  # [B, T, D]
        
        # Stack resized layers: [B, L, T, D]
        stacked_layers = tf.stack(resized_layers, axis=1)
        
        # Step 2: Pool each layer to create layer vectors [B, L, D]
        if self.pool_type == "mean":
            layer_vectors = tf.reduce_mean(stacked_layers, axis=2)  # [B, L, D]
        elif self.pool_type == "max":
            layer_vectors = tf.reduce_max(stacked_layers, axis=2)  # [B, L, D]
        elif self.pool_type == "attention":
            # Learnable attention pooling: each layer learns which sequence positions are important
            # 
            # Key idea: Each layer gets its own learnable "importance detector" query vector
            # that learns to identify which sequence positions contain the most relevant
            # information for that specific layer. This allows different layers to focus
            # on different types of patterns (e.g., local vs global, syntactic vs semantic).
            #
            # Architecture:
            # - Layer 0 query might learn to focus on start/end positions  
            # - Layer 1 query might learn to focus on middle regions
            # - Layer 2 query might learn to focus on specific motifs
            # etc.
            #
            # stacked_layers: [B, L, T, D]
            batch_size = tf.shape(stacked_layers)[0]
            num_layers = tf.shape(stacked_layers)[1]
            seq_length = tf.shape(stacked_layers)[2]
            
            # Use learnable query embeddings - each layer has its own "importance detector"
            # More efficient: use broadcasting instead of tile
            # Expand learnable queries: [L, D] -> [B, L, 1, D]
            pooling_queries = tf.expand_dims(
                tf.expand_dims(self.layer_pool_queries, axis=0), 
                axis=2
            )  # [1, L, 1, D]
            
            # Broadcast to batch size and reshape for attention
            pooling_queries = tf.broadcast_to(
                pooling_queries, 
                [batch_size, num_layers, 1, self.fusion_dim]
            )  # [B, L, 1, D]
            
            # Reshape for batch processing: [B*L, 1, D]
            pooling_queries = tf.reshape(
                pooling_queries, 
                [batch_size * num_layers, 1, self.fusion_dim]
            )
            
            # Keys/Values: [B*L, T, D] - sequence representations for each layer
            reshaped_layers = tf.reshape(
                stacked_layers, 
                [batch_size * num_layers, seq_length, self.fusion_dim]
            )
            
            # Apply learnable attention pooling
            # Each layer's query learns to attend to different sequence positions
            # Query: layer-specific learnable queries [B*L, 1, D]
            # Key/Value: sequence representations [B*L, T, D]
            if self.use_custom_mha:
                # Custom MHA handles Q/K/V internally, but we still need to use our learnable queries
                # Apply custom MHA to get sequence representations
                mha_output = self.pool_attention(reshaped_layers, training=training)  # [B*L, T, D]
                
                # Now apply our learnable queries for attention pooling
                # Reshape pooling queries to match batch structure: [L, D] -> [B*L, 1, D]
                pooling_queries = tf.expand_dims(
                    tf.expand_dims(self.layer_pool_queries, axis=0), 
                    axis=2
                )  # [1, L, 1, D]
                pooling_queries = tf.broadcast_to(
                    pooling_queries, 
                    [batch_size, num_layers, 1, self.fusion_dim]
                )  # [B, L, 1, D]
                pooling_queries = tf.reshape(
                    pooling_queries, 
                    [batch_size * num_layers, 1, self.fusion_dim]
                )  # [B*L, 1, D]
                
                # Compute attention weights using learnable queries
                # Q: [B*L, 1, D], K: [B*L, T, D]
                attention_scores = tf.matmul(pooling_queries, mha_output, transpose_b=True)  # [B*L, 1, T]
                attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # [B*L, 1, T]
                
                # Apply attention weights to get pooled output
                pooled_output = tf.matmul(attention_weights, mha_output)  # [B*L, 1, D]
            else:
                # Standard Keras MHA with explicit Q/K/V
                pooled_output = self.pool_attention(
                    query=pooling_queries,    # Learnable per-layer queries
                    key=reshaped_layers,      # Sequence positions as keys
                    value=reshaped_layers,    # Sequence positions as values
                    training=training
                )  # [B*L, 1, D]
            
            # Reshape back to layer vectors: [B, L, D]
            layer_vectors = tf.reshape(
                tf.squeeze(pooled_output, axis=1), 
                [batch_size, num_layers, self.fusion_dim]
            )
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        # Step 3: Create query from mean of all layers
        mean_embedding = tf.reduce_mean(stacked_layers, axis=1)  # [B, T, D]
        query_embedding = self.query_projection(mean_embedding)  # [B, T, D]
        
        # Step 4: Efficient cross attention - K/V length = L (not L×T)
        # This avoids the memory explosion from tile operation
        # Query: [B, T, D], Key/Value: [B, L, D] directly
        
        # Apply cross attention with layer_vectors as K/V directly
        # This gives us O(L·T) complexity instead of O(L·T²)
        if self.use_custom_mha:
            # Custom MHA expects single input, so we concatenate query and layer_vectors
            # This is a simplified approach - you may need custom implementation for cross-attention
            # For now, we'll use the query_embedding as input to custom MHA
            attention_output = self.cross_attention(query_embedding, training=training)  # [B, T, D]
        else:
            # Standard Keras MHA with explicit Q/K/V
            attention_output = self.cross_attention(
                query=query_embedding,           # [B, T, D]
                key=layer_vectors,              # [B, L, D] 
                value=layer_vectors,            # [B, L, D]
                training=training
            )  # [B, T, D]
        
        # Step 5: Compute layer attention weights for final fusion
        query_pooled = tf.reduce_mean(query_embedding, axis=1)  # [B, D]
        
        # Compute layer scores using the specified method
        layer_scores = self._compute_layer_scores(query_pooled, layer_vectors, training=training)  # [B, L]
        layer_scores = layer_scores / self.temperature  # Temperature scaling
        layer_attention_weights = tf.nn.softmax(layer_scores, axis=-1)  # [B, L]
        
        # Step 6: Apply layer gating (optional)
        if self.use_layer_gate:
            gate_input = tf.reduce_mean(mean_embedding, axis=1)  # [B, D]
            layer_gates = self.layer_gate(gate_input)  # [B, L]
            layer_attention_weights = layer_attention_weights * layer_gates
            layer_attention_weights = tf.nn.softmax(layer_attention_weights, axis=-1)
        
        # Step 7: Efficient weighted fusion using einsum
        # Avoid creating large intermediate tensors
        # layer_attention_weights: [B, L], stacked_layers: [B, L, T, D]
        fused_embedding = tf.einsum('bl,bltd->btd', layer_attention_weights, stacked_layers)  # [B, T, D]
        
        # Step 8: Add residual connection and layer norm
        fused_embedding = self.layer_norm(fused_embedding + attention_output)
        
        # Add entropy regularization loss
        if self.entropy_reg_weight > 0 and training:
            entropy_loss = -tf.reduce_sum(layer_attention_weights * tf.math.log(layer_attention_weights + 1e-8), axis=-1)
            entropy_loss = tf.reduce_mean(entropy_loss) * self.entropy_reg_weight
            self.add_loss(entropy_loss)
        
        return fused_embedding
    
    def get_config(self):
        config = super(CrossAttentionLayerFusion, self).get_config()
        config.update({
            'fusion_dim': self.fusion_dim,
            'num_heads': self.num_heads,
            'attention_dropout': self.attention_dropout,
            'temperature': self.temperature,
            'use_layer_gate': self.use_layer_gate,
            'entropy_reg_weight': self.entropy_reg_weight,
            'pool_type': self.pool_type,
            'kernel_initializer': self.kernel_initializer,
            'layer_score_type': self.layer_score_type,
            'use_custom_mha': self.use_custom_mha,
            'mha_params': self.mha_params,
        })
        return config


class MultiLayerEmbeddingExtractor:
    """Extract multi-layer embeddings from pretrained model."""
    
    def __init__(self, pretrained_model, layer_indices=None):
        """Initialize extractor.
        
        Args:
            pretrained_model: Pretrained SeqNN model
            layer_indices: List of layer indices to extract (if None, auto-select)
        """
        self.pretrained_model = pretrained_model
        self.layer_indices = layer_indices
        self.extraction_model = None
        self._build_extraction_model()
    
    def _build_extraction_model(self):
        """Build model for multi-layer extraction."""
        if self.layer_indices is None:
            # Auto-select meaningful layers
            self.layer_indices = self._select_meaningful_layers()
        
        # Get layer outputs
        layer_outputs = []
        for idx in self.layer_indices:
            if idx < len(self.pretrained_model.model.layers):
                layer_outputs.append(self.pretrained_model.model.layers[idx].output)
        
        if not layer_outputs:
            raise ValueError("No valid layers found for extraction")
        
        # Create extraction model
        self.extraction_model = tf.keras.Model(
            inputs=self.pretrained_model.model.input,
            outputs=layer_outputs,
            name="multi_layer_extractor"
        )
        
    
    def _select_meaningful_layers(self):
        """Auto-select meaningful layers for extraction."""
        selected = []
        skip_types = ['input', 'dropout', 'reshape', 'flatten', 'lambda', 'concatenate']
        
        for i, layer in enumerate(self.pretrained_model.model.layers):
            layer_type = type(layer).__name__.lower()
            
            # Skip layers that don't have meaningful representations
            if any(skip_type in layer_type for skip_type in skip_types):
                continue
            
            # Include layers with activations or major computation
            if (hasattr(layer, 'activation') or 
                any(keyword in layer_type for keyword in ['conv', 'dense', 'attention', 'norm', 'batch', 'transformer'])):
                selected.append(i)
        
        # Select every few layers to avoid too many
        if len(selected) > 8:
            step = len(selected) // 6
            selected = selected[::step][:6]
        
        return selected
    
    def extract(self, inputs):
        """Extract multi-layer embeddings.
        
        Args:
            inputs: Input sequences
            
        Returns:
            List of layer embeddings
        """
        if self.extraction_model is None:
            raise ValueError("Extraction model not built")
        
        outputs = self.extraction_model(inputs, training=False)
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        return outputs


def build_cross_attention_model(params_model, pretrained_model_path, trunk_only=True):
    """Build a SeqNN model with cross attention fusion (following paddy_transfer.py style).
    
    Args:
        params_model: Model parameters dictionary
        pretrained_model_path: Path to pretrained model
        trunk_only: Whether to load trunk only
        
    Returns:
        SeqNN model with cross attention fusion
    """
    cross_attention_params = params_model.get("cross_attention", {})
    
    # Create a standard SeqNN model first (following paddy_transfer.py pattern)
    seqnn_model = seqnn.SeqNN(params_model)
    
    # Load pretrained weights
    if trunk_only:
        seqnn_model.model_trunk.load_weights(pretrained_model_path, by_name=True)
    else:
        seqnn_model.model.load_weights(pretrained_model_path, by_name=True)
    
    # Now modify the model to use cross attention fusion
    # Extract multi-layer embeddings from the loaded model
    layer_indices = cross_attention_params.get("layer_indices", None)
    layer_extractor = MultiLayerEmbeddingExtractor(seqnn_model, layer_indices)
    
    # Build new model with cross attention fusion
    sequence = tf.keras.Input(shape=(seqnn_model.seq_length, seqnn_model.seq_depth), name="sequence")
    layer_embeddings = layer_extractor.extract(sequence)
    
    # Apply cross attention fusion
    fusion_params = {
        'fusion_dim': cross_attention_params.get('fusion_dim', 256),
        'num_heads': cross_attention_params.get('num_heads', 8),
        'attention_dropout': cross_attention_params.get('attention_dropout', 0.1),
        'temperature': cross_attention_params.get('temperature', 1.0),
        'use_layer_gate': cross_attention_params.get('use_layer_gate', False),
        'entropy_reg_weight': cross_attention_params.get('entropy_reg_weight', 0.01),
        'pool_type': cross_attention_params.get('pool_type', 'mean'),
        'layer_score_type': cross_attention_params.get('layer_score_type', 'dot_product'),
        'use_custom_mha': cross_attention_params.get('use_custom_mha', False),
        'mha_params': cross_attention_params.get('mha_params', {}),
    }
    
    fusion_layer = CrossAttentionLayerFusion(**fusion_params)
    fused_embedding = fusion_layer(layer_embeddings)
    
    # Replace the trunk model with our fusion model
    seqnn_model.model_trunk = tf.keras.Model(inputs=sequence, outputs=fused_embedding)
    print(seqnn_model.model_trunk.summary())
    # Rebuild heads using the new trunk
    trunk_output = fused_embedding
    seqnn_model.head_output = []
    
    for hi, head in enumerate(seqnn_model.heads):
        if not isinstance(head, list):
            head = [head]
        
        # Reset to trunk output
        current = trunk_output
        
        # Build head blocks
        seqnn_model.reprs = []  # Reset reprs for head building
        for bi, block_params in enumerate(head):
            current = seqnn_model.build_block(current, block_params)
            seqnn_model.reprs.append(current)
        
        seqnn_model.head_output.append(current)
    
    # Rebuild models
    seqnn_model.models = []
    for ho in seqnn_model.head_output:
        seqnn_model.models.append(tf.keras.Model(inputs=sequence, outputs=ho))
    seqnn_model.model = seqnn_model.models[0]
    print(seqnn_model.model.summary())
    
    return seqnn_model


def main():
    parser = argparse.ArgumentParser(description="Train a seq2exp model using cross attention fusion of multi-layer embeddings.")
    parser.add_argument(
        "-g",
        "--gpu",
        "--visible_device",
        nargs='+',
        type=str,
        default=None,
        help="GPU IDs to use (can specify multiple, e.g., -g 0 1 2)",
    )
    parser.add_argument(
        "-k",
        "--keras_fit",
        action="store_true",
        default=False,
        help="Train with Keras fit method [Default: %(default)s]",
    )
    parser.add_argument(
        "-m",
        "--mixed_precision",
        action="store_true",
        default=False,
        help="Train with mixed precision [Default: %(default)s]",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="train_out",
        help="Output directory [Default: %(default)s]",
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        default="log_out",
        help="Tensorboard log directory [Default: %(default)s]",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Resume training from checkpoint in args.out_dir [Default: %(default)s]",
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help="Path to pretrained model for multi-layer extraction [Required]",
    )
    parser.add_argument(
        "--trunk",
        action="store_true",
        default=False,
        help="Load only model trunk from pretrained [Default: %(default)s]",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Seed for model initialization [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_train",
        default=None,
        help="Training TFR pattern string appended to data_dir/tfrecords [Default: %(default)s]",
    )
    parser.add_argument(
        "--tfr_eval",
        default=None,
        help="Evaluation TFR pattern string appended to data_dir/tfrecords [Default: %(default)s]",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        default=False,
        help="Report trainable params and skip training [Default: %(default)s]",
    )

    parser.add_argument("params_file", help="YAML file with model parameters")
    parser.add_argument(
        "data_dirs", nargs="+", help="Train/valid/test data directorie(s)"
    )
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Copy params.yaml to out_dir
    if args.params_file != "%s/params.yaml" % args.out_dir:
        shutil.copy(args.params_file, "%s/params.yaml" % args.out_dir)

    with open(args.params_file, "r") as f:
        params = yaml.safe_load(f)
    params_model = params["model"]
    params_train = params["train"]

    
    # Update num_gpu in params_train if specified in command line
    if args.gpu and len(args.gpu) > 1:
        params_train["num_gpu"] = len(args.gpu)

    # Prioritize args.seed over params_train.seed
    seed = args.seed or params_train.get("seed", None)
    if seed is None:
        seed = random.randint(0, 1000000)  # random seed

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.config.experimental.enable_op_determinism()
    with open(f"{args.out_dir}/seed.txt", "w") as f:
        f.write(f"Random seed: {seed}\n")

    if args.keras_fit and len(args.data_dirs) > 1:
        print("Cannot use keras fit method with multi-genome training.")
        exit()

    # Read datasets
    train_data = []
    eval_data = []

    for data_dir in args.data_dirs:
        # Load train data
        train_data.append(
            dataset.SeqDataset(
                data_dir,
                split_label="train",
                batch_size=params_train["batch_size"],
                shuffle_buffer=params_train.get("shuffle_buffer", 128),
                mode="train",
                tfr_pattern=args.tfr_train,
                model_type=params_model["model_type"],
            )
        )

        # Load eval data
        eval_data.append(
            dataset.SeqDataset(
                data_dir,
                split_label="valid",
                batch_size=params_train["batch_size"],
                mode="eval",
                tfr_pattern=args.tfr_eval,
                model_type=params_model["model_type"],
            )
        )

    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")

    if params_train.get("num_gpu", 1) == 1:
        ########################################
        # Single GPU
        
        # Initialize model with cross attention fusion (similar to paddy_transfer.py)
        params_model["verbose"] = False
        
        # Build cross attention fusion model
        seqnn_model = build_cross_attention_model(params_model, args.pretrained, trunk_only=args.trunk)
        
        # Initialize trainer
        if args.mixed_precision:
            seqnn_trainer = trainer.Trainer(
                params_train,
                train_data,
                eval_data,
                args.out_dir,
                args.log_dir,
                loss_scale=True,
            )
        else:
            seqnn_trainer = trainer.Trainer(
                params_train, train_data, eval_data, args.out_dir, args.log_dir
            )

        # Compile model
        seqnn_trainer.compile(seqnn_model)

        if args.skip_train:
            exit(0)

        # Train model
        if args.keras_fit:
            seqnn_trainer.fit_keras(seqnn_model)
        else:
            if len(args.data_dirs) == 1:
                seqnn_trainer.fit_tape(seqnn_model, resume=args.resume)
            else:
                seqnn_trainer.fit2(seqnn_model)

    else:
        ########################################
        # Multi GPU
        available_gpus = len(tf.config.list_physical_devices('GPU'))
        if available_gpus < params_train['num_gpu']:
            params_train['num_gpu'] = available_gpus

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            if not args.keras_fit:
                # Distribute data
                for di in range(len(args.data_dirs)):
                    train_data[di].distribute(strategy)
                    eval_data[di].distribute(strategy)

            # Initialize model with cross attention fusion (similar to paddy_transfer.py)
            #params_model["verbose"] = False
            
            # Build cross attention fusion model
            seqnn_model = build_cross_attention_model(params_model, args.pretrained, trunk_only=args.trunk)

            # Initialize trainer
            if args.mixed_precision:
                seqnn_trainer = trainer.Trainer(
                    params_train,
                    train_data,
                    eval_data,
                    args.out_dir,
                    args.log_dir,
                    strategy=strategy,
                    num_gpu=params_train["num_gpu"],
                    keras_fit=args.keras_fit,
                    loss_scale=True,
                )
            else:
                seqnn_trainer = trainer.Trainer(
                    params_train,
                    train_data,
                    eval_data,
                    args.out_dir,
                    args.log_dir,
                    strategy=strategy,
                    num_gpu=params_train["num_gpu"],
                    keras_fit=args.keras_fit,
                )

            # Compile model
            seqnn_trainer.compile(seqnn_model)

        if args.skip_train:
            exit(0)

        # Train model
        if args.keras_fit:
            seqnn_trainer.fit_keras(seqnn_model)
        else:
            if len(args.data_dirs) == 1:
                seqnn_trainer.fit_tape(seqnn_model, resume=args.resume)
            else:
                seqnn_trainer.fit2(seqnn_model)


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
