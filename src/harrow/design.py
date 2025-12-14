# design.py
# Sequence design/optimization
# Simplified implementation for Paddy/Harrow
# Author: Adapted for Paddy/Harrow

import numpy
import tensorflow as tf


def design(model, X_init, objective_fn, n_iter=100, learning_rate=0.1,
          temperature=1.0, verbose=False):
    """Design sequences to optimize an objective function.
    
    This function uses gradient-based optimization to design sequences that
    maximize a given objective function.
    
    Parameters
    ----------
    model: paddy.seqnn.SeqNN
        A Paddy SeqNN model.
    
    X_init: numpy.ndarray, shape=(-1, length, alphabet_size)
        Initial sequences to optimize from.
    
    objective_fn: function
        A function that takes model predictions and returns a scalar loss to minimize.
    
    n_iter: int, optional
        Number of optimization iterations. Default is 100.
    
    learning_rate: float, optional
        Learning rate for optimization. Default is 0.1.
    
    temperature: float, optional
        Temperature for softmax relaxation. Default is 1.0.
    
    verbose: bool, optional
        Whether to print progress. Default is False.
    
    Returns
    -------
    X_opt: numpy.ndarray
        Optimized sequences (one-hot encoded).
    
    losses: list
        Loss values at each iteration.
    """
    
    # Convert to continuous representation for optimization
    X_logits = tf.Variable(numpy.log(X_init + 1e-8), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    losses = []
    for i in range(n_iter):
        with tf.GradientTape() as tape:
            # Softmax relaxation
            X_soft = tf.nn.softmax(X_logits / temperature, axis=-1)
            
            # Forward pass
            y_pred = model.model(X_soft, training=False)
            
            # Compute loss
            loss = objective_fn(y_pred)
        
        # Backward pass
        grads = tape.gradient(loss, [X_logits])
        optimizer.apply_gradients(zip(grads, [X_logits]))
        
        losses.append(float(loss))
        
        if verbose and (i % 10 == 0 or i == n_iter - 1):
            print(f"Iteration {i+1}/{n_iter}, Loss: {loss:.4f}")
    
    # Convert back to one-hot
    X_opt_soft = tf.nn.softmax(X_logits, axis=-1).numpy()
    X_opt = numpy.zeros_like(X_opt_soft)
    X_opt[numpy.arange(X_opt.shape[0])[:, None], 
          numpy.arange(X_opt.shape[1]),
          X_opt_soft.argmax(axis=-1)] = 1
    
    return X_opt, losses

