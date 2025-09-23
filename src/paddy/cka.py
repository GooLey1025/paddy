"""
Centered Kernel Alignment (CKA) implementation for TensorFlow models.

This module provides functions to compute CKA similarity between neural network
representations, including both intra-model and inter-model comparisons.

Based on the paper:
"Similarity of Neural Network Representations Revisited" by Kornblith et al.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import h5py
import gc



def clear_memory():
    """Clear memory and GPU cache."""
    gc.collect()


def linear_kernel(x: tf.Tensor) -> tf.Tensor:
    """Compute linear kernel (Gram matrix) K = X @ X^T.
    
    Args:
        x: Input tensor of shape [n, d] where n is number of samples, d is feature dimension
        
    Returns:
        Kernel matrix of shape [n, n]
    """
    return tf.linalg.matmul(x, x, transpose_b=True)


def rbf_kernel(x: tf.Tensor, sigma: float = 1.0) -> tf.Tensor:
    """Compute RBF (Gaussian) kernel.
    
    Args:
        x: Input tensor of shape [n, d]
        sigma: RBF kernel bandwidth parameter
        
    Returns:
        RBF kernel matrix of shape [n, n]
    """
    # Compute pairwise squared distances
    x_norm_sq = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
    distances_sq = x_norm_sq + tf.transpose(x_norm_sq) - 2 * tf.linalg.matmul(x, x, transpose_b=True)
    
    # Use median distance as bandwidth if sigma is not specified
    if sigma == 'auto':
        median_dist_sq = tf.nn.top_k(tf.reshape(distances_sq, [-1]), 
                                    k=tf.size(distances_sq)//2).values[-1]
        sigma = tf.sqrt(median_dist_sq)
    
    return tf.exp(-distances_sq / (2 * sigma ** 2))


def center_gram_matrix(K: tf.Tensor) -> tf.Tensor:
    """Center a Gram matrix K using the centering matrix H = I - (1/n)11^T.
    
    Args:
        K: Gram matrix of shape [n, n]
        
    Returns:
        Centered Gram matrix
    """
    n = tf.shape(K)[0]
    n_float = tf.cast(n, K.dtype)
    
    # Centering matrix H = I - (1/n)11^T
    ones = tf.ones([n, 1], dtype=K.dtype)
    H = tf.eye(n, dtype=K.dtype) - tf.linalg.matmul(ones, ones, transpose_b=True) / n_float
    
    # Apply centering: HKH
    return tf.linalg.matmul(tf.linalg.matmul(H, K), H)


def hsic(K: tf.Tensor, L: tf.Tensor) -> tf.Tensor:
    """Compute Hilbert-Schmidt Independence Criterion (HSIC).
    
    HSIC(K, L) = (1/(n-1)^2) * tr(KHLH)
    
    Args:
        K: First kernel matrix of shape [n, n]
        L: Second kernel matrix of shape [n, n]
        
    Returns:
        HSIC value (scalar tensor)
    """
    n = tf.shape(K)[0]
    n_float = tf.cast(n, K.dtype)
    
    # Center both kernel matrices
    K_centered = center_gram_matrix(K)
    L_centered = center_gram_matrix(L)
    
    # Compute HSIC = tr(K_centered @ L_centered) / (n-1)^2
    hsic_value = tf.linalg.trace(tf.linalg.matmul(K_centered, L_centered))
    hsic_value = hsic_value / ((n_float - 1) ** 2)
    
    return hsic_value


def cka_numpy(X: np.ndarray, Y: np.ndarray, kernel: str = 'linear') -> float:
    """Compute Centered Kernel Alignment (CKA) between two representations using NumPy.
    
    CKA(X, Y) = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
    
    Args:
        X: First representation array of shape [n, d1]
        Y: Second representation array of shape [n, d2]
        kernel: Kernel type ('linear' only supported for numpy version)
        
    Returns:
        CKA similarity score (scalar between 0 and 1)
    """
    if kernel != 'linear':
        raise ValueError("NumPy CKA currently only supports linear kernel")
    
    # Ensure float32 for consistency
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    
    n, d1 = X.shape
    d2 = Y.shape[1]
    
    
    # For very high-dimensional features, use sample-space computation instead
    # This avoids creating huge d×d matrices
    max_feature_dim = 50000  # Threshold for switching to sample-space computation
    
    if d1 > max_feature_dim or d2 > max_feature_dim:
        return cka_numpy_sample_space(X, Y)
    else:
        return cka_numpy_feature_space(X, Y)


def cka_numpy_feature_space(X: np.ndarray, Y: np.ndarray) -> float:
    """CKA computation in feature space (efficient for low-dimensional features)."""
    n = X.shape[0]
    
    # Center the data
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    Y_centered = Y - np.mean(Y, axis=0, keepdims=True)
    
    # Compute cross-covariance matrices (feature space)
    XY = np.dot(X_centered.T, Y_centered)  # [d1, d2]
    XX = np.dot(X_centered.T, X_centered)  # [d1, d1]
    YY = np.dot(Y_centered.T, Y_centered)  # [d2, d2]
    
    # Compute HSIC values using Frobenius norm
    hsic_XY = np.trace(np.dot(XY, XY.T))
    hsic_XX = np.trace(np.dot(XX, XX))
    hsic_YY = np.trace(np.dot(YY, YY))
    
    # Normalize by (n-1)^2 to match standard HSIC formula
    normalizer = (n - 1) ** 2
    hsic_XY = hsic_XY / normalizer
    hsic_XX = hsic_XX / normalizer  
    hsic_YY = hsic_YY / normalizer
    
    cka_value = hsic_XY / np.sqrt(hsic_XX * hsic_YY + 1e-12)
    
    return float(cka_value)


def cka_numpy_sample_space(X: np.ndarray, Y: np.ndarray) -> float:
    """CKA computation in sample space (efficient for high-dimensional features)."""
    n = X.shape[0]
    
    # Center the data
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    Y_centered = Y - np.mean(Y, axis=0, keepdims=True)
    
    # Compute kernel matrices in sample space (n×n instead of d×d)
    K_X = np.dot(X_centered, X_centered.T)  # [n, n]
    K_Y = np.dot(Y_centered, Y_centered.T)  # [n, n]
    
    # Center the kernel matrices
    ones = np.ones((n, 1), dtype=np.float32)
    H = np.eye(n, dtype=np.float32) - np.dot(ones, ones.T) / n
    
    K_X_centered = np.dot(np.dot(H, K_X), H)
    K_Y_centered = np.dot(np.dot(H, K_Y), H)
    
    # Compute HSIC values
    hsic_XY = np.trace(np.dot(K_X_centered, K_Y_centered))
    hsic_XX = np.trace(np.dot(K_X_centered, K_X_centered))
    hsic_YY = np.trace(np.dot(K_Y_centered, K_Y_centered))
    
    # Normalize by (n-1)^2
    normalizer = (n - 1) ** 2
    hsic_XY = hsic_XY / normalizer
    hsic_XX = hsic_XX / normalizer
    hsic_YY = hsic_YY / normalizer
    
    # CKA computation
    cka_value = hsic_XY / np.sqrt(hsic_XX * hsic_YY + 1e-12)
    
    return float(cka_value)





@tf.function
def cka_gpu(X, Y, kernel: str = 'linear', sigma: float = 1.0):
    """GPU-accelerated CKA computation using TensorFlow operations.
    
    Args:
        X: First representation tensor of shape [n, d1]
        Y: Second representation tensor of shape [n, d2]
        kernel: Kernel type ('linear' or 'rbf')
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
        
    Returns:
        CKA similarity score (scalar between 0 and 1)
    """
    # Ensure float32 for GPU efficiency
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)
    
    # Compute kernel matrices
    if kernel == 'linear':
        K_X = linear_kernel(X)
        K_Y = linear_kernel(Y)
    elif kernel == 'rbf':
        K_X = rbf_kernel(X, sigma)
        K_Y = rbf_kernel(Y, sigma)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")
    
    # Compute HSIC values
    hsic_XY = hsic(K_X, K_Y)
    hsic_XX = hsic(K_X, K_X)
    hsic_YY = hsic(K_Y, K_Y)
    
    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    cka_value = hsic_XY / tf.sqrt(hsic_XX * hsic_YY + 1e-8)
    
    return cka_value


@tf.function
def batched_hsic_gpu(K_batch: tf.Tensor, L_batch: tf.Tensor, epsilon: float = 1e-4) -> tf.Tensor:
    """GPU-accelerated batched HSIC computation inspired by PyTorch implementation.
    
    Args:
        K_batch: Batch of kernel matrices of shape [batch_size, n, n]
        L_batch: Batch of kernel matrices of shape [batch_size, n, n]
        epsilon: Small regularization value
        
    Returns:
        HSIC values for each pair in batch [batch_size]
    """
    batch_size, n, _ = tf.shape(K_batch)[0], tf.shape(K_batch)[1], tf.shape(K_batch)[2]
    n_float = tf.cast(n, K_batch.dtype)
    
    # Set diagonals to zero (K~, L~) - inspired by PyTorch implementation
    diag_mask = tf.eye(n, dtype=K_batch.dtype)
    diag_mask = tf.expand_dims(diag_mask, 0)  # [1, n, n]
    
    K_tilde = K_batch * (1 - diag_mask)
    L_tilde = L_batch * (1 - diag_mask)
    
    # Compute KL
    KL = tf.linalg.matmul(K_tilde, L_tilde)
    
    # Main term: trace(KL)
    trace_KL = tf.linalg.trace(KL)  # [batch_size]
    
    # Middle term: sum(K) * sum(L) / ((n-1) * (n-2))
    sum_K = tf.reduce_sum(K_tilde, axis=[1, 2])  # [batch_size]
    sum_L = tf.reduce_sum(L_tilde, axis=[1, 2])  # [batch_size]
    middle_term = (sum_K * sum_L) / ((n_float - 1) * (n_float - 2))
    
    # Right term: 2 * sum(KL) / (n-2)
    sum_KL = tf.reduce_sum(KL, axis=[1, 2])  # [batch_size]
    right_term = 2 * sum_KL / (n_float - 2)
    
    # Final HSIC computation
    main_term = trace_KL + middle_term - right_term
    hsic_values = main_term / (n_float ** 2 - 3 * n_float)
    
    return hsic_values * epsilon


def cka(X, Y, kernel: str = 'linear', sigma: float = 1.0):
    """Compute Centered Kernel Alignment (CKA) between two representations.
    
    CKA(X, Y) = HSIC(K_X, K_Y) / sqrt(HSIC(K_X, K_X) * HSIC(K_Y, K_Y))
    
    Args:
        X: First representation tensor/array of shape [n, d1]
        Y: Second representation tensor/array of shape [n, d2]
        kernel: Kernel type ('linear' or 'rbf')
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
        
    Returns:
        CKA similarity score (scalar between 0 and 1)
    """
    # Convert numpy arrays to tensors if needed
    if isinstance(X, np.ndarray):
        X = tf.constant(X, dtype=tf.float32)
    if isinstance(Y, np.ndarray):
        Y = tf.constant(Y, dtype=tf.float32)
    
    # Use GPU-accelerated version
    return cka_gpu(X, Y, kernel, sigma)


class GPUCKAAnalyzer:
    """GPU-accelerated CKA analyzer inspired by PyTorch implementation."""
    
    def __init__(self, kernel: str = 'linear', batch_size: int = 1000, 
                 max_samples_for_cka: int = 1500, group_size: int = 64, epsilon: float = 1e-4):
        """Initialize GPU-accelerated CKA analyzer.
        
        Args:
            kernel: Kernel type ('linear' or 'rbf')
            batch_size: Batch size for data processing
            max_samples_for_cka: Maximum number of samples for CKA computation
            group_size: Group size for batched GPU operations (inspired by PyTorch version)
            epsilon: Small regularization value for numerical stability
        """
        self.kernel = kernel
        self.batch_size = batch_size
        self.max_samples_for_cka = max_samples_for_cka
        self.group_size = group_size
        self.epsilon = epsilon
        
        print(f"GPU CKA Analyzer initialized:")
        print(f"  - Kernel: {kernel}")
        print(f"  - Group size: {group_size}")
        print(f"  - Epsilon: {epsilon}")
    
    def extract_activations_gpu(self, model: tf.keras.Model, data: tf.Tensor, 
                               layer_names: List[str]) -> Dict[str, tf.Tensor]:
        """GPU-optimized activation extraction with memory management.
        
        Args:
            model: TensorFlow model
            data: Input data tensor
            layer_names: Layer names to extract
            
        Returns:
            Dictionary of layer activations
        """
        print(f"GPU-accelerated extraction from {len(layer_names)} layers")
        
        # Create extraction model once
        layer_outputs = []
        valid_layer_names = []
        
        for layer_name in layer_names:
            try:
                layer = model.get_layer(layer_name)
                layer_outputs.append(layer.output)
                valid_layer_names.append(layer_name)
            except ValueError:
                print(f"Warning: Layer '{layer_name}' not found")
                continue
        
        if not layer_outputs:
            raise ValueError("No valid layers found")
        
        extraction_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
        
        # Process data in chunks to avoid GPU memory issues
        total_samples = tf.shape(data)[0].numpy()
        chunk_size = min(self.batch_size * 4, total_samples)  # Larger chunks for efficiency
        
        print(f"Processing {total_samples} samples in chunks of {chunk_size}")
        
        result = {}
        first_chunk = True
        
        for start_idx in tqdm(range(0, total_samples, chunk_size), desc="GPU extraction"):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_data = data[start_idx:end_idx]
            
            # Extract activations for this chunk
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                chunk_activations = extraction_model(chunk_data, training=False)
                if not isinstance(chunk_activations, list):
                    chunk_activations = [chunk_activations]
            
            # Initialize result tensors on first chunk
            if first_chunk:
                for i, activation in enumerate(chunk_activations):
                    layer_name = valid_layer_names[i]
                    # Flatten spatial dimensions
                    if len(activation.shape) > 2:
                        flat_dim = np.prod(activation.shape[1:])
                        result[layer_name] = tf.Variable(
                            tf.zeros([total_samples, flat_dim], dtype=tf.float32),
                            trainable=False
                        )
                    else:
                        result[layer_name] = tf.Variable(
                            tf.zeros([total_samples, activation.shape[1]], dtype=tf.float32),
                            trainable=False
                        )
                first_chunk = False
            
            # Store chunk activations
            for i, activation in enumerate(chunk_activations):
                layer_name = valid_layer_names[i]
                
                # Flatten if needed
                if len(activation.shape) > 2:
                    flattened = tf.reshape(activation, [end_idx - start_idx, -1])
                else:
                    flattened = activation
                
                # Update the variable
                result[layer_name][start_idx:end_idx].assign(flattened)
        
        # Convert Variables to Tensors
        for layer_name in result:
            result[layer_name] = tf.identity(result[layer_name])
        
        return result
    
    def compute_cka_matrix_gpu_batched(self, activations1: Dict[str, tf.Tensor], 
                                      activations2: Dict[str, tf.Tensor] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """GPU-accelerated CKA matrix computation with batched operations.
        
        Args:
            activations1: First model activations
            activations2: Second model activations (None for intra-model)
            
        Returns:
            CKA matrix, layer names 1, layer names 2
        """
        is_intra_model = (activations2 is None)
        if activations2 is None:
            activations2 = activations1
        
        layer_names1 = list(activations1.keys())
        layer_names2 = list(activations2.keys())
        
        n_layers1 = len(layer_names1)
        n_layers2 = len(layer_names2)
        
        print(f"Computing GPU-accelerated CKA matrix ({n_layers1} x {n_layers2})")
        print(f"Using group size: {self.group_size}")
        
        # Initialize result matrix
        cka_matrix = np.zeros((n_layers1, n_layers2), dtype=np.float32)
        
        # Pre-compute all kernel matrices for efficiency
        print("Pre-computing kernel matrices...")
        kernels1 = {}
        kernels2 = {}
        
        for name, activation in tqdm(activations1.items(), desc="Computing kernels 1"):
            if self.kernel == 'linear':
                kernels1[name] = linear_kernel(activation)
            elif self.kernel == 'rbf':
                kernels1[name] = rbf_kernel(activation)
        
        if not is_intra_model:
            for name, activation in tqdm(activations2.items(), desc="Computing kernels 2"):
                if self.kernel == 'linear':
                    kernels2[name] = linear_kernel(activation)
                elif self.kernel == 'rbf':
                    kernels2[name] = rbf_kernel(activation)
        else:
            kernels2 = kernels1
        
        # Compute CKA in batches for GPU efficiency
        total_pairs = n_layers1 * n_layers2
        if is_intra_model:
            total_pairs = n_layers1 * (n_layers1 + 1) // 2  # Upper triangle only
        
        print(f"Computing {total_pairs} CKA pairs in batches of {self.group_size}")
        
        pairs_processed = 0
        with tqdm(total=total_pairs, desc="Computing CKA pairs") as pbar:
            
            for i in range(n_layers1):
                start_j = i if is_intra_model else 0
                
                for j_start in range(start_j, n_layers2, self.group_size):
                    j_end = min(j_start + self.group_size, n_layers2)
                    batch_size_current = j_end - j_start
                    
                    if batch_size_current == 0:
                        continue
                    
                    # Prepare batch of kernel matrices
                    K_batch = tf.stack([kernels1[layer_names1[i]]] * batch_size_current)
                    L_batch = tf.stack([kernels2[layer_names2[j]] for j in range(j_start, j_end)])
                    
                    # Compute batched HSIC
                    hsic_XY_batch = batched_hsic_gpu(K_batch, L_batch, self.epsilon)
                    hsic_XX_batch = batched_hsic_gpu(K_batch, K_batch, self.epsilon)
                    hsic_YY_batch = batched_hsic_gpu(L_batch, L_batch, self.epsilon)
                    
                    # Compute CKA values
                    cka_batch = hsic_XY_batch / tf.sqrt(hsic_XX_batch * hsic_YY_batch + 1e-8)
                    cka_batch_np = cka_batch.numpy()
                    
                    # Store results
                    for idx, j in enumerate(range(j_start, j_end)):
                        cka_matrix[i, j] = cka_batch_np[idx]
                        if is_intra_model and i != j:
                            cka_matrix[j, i] = cka_batch_np[idx]
                        pairs_processed += 1
                        pbar.update(1)
                
                # Clear GPU memory periodically
                if i % 5 == 0:
                    clear_memory()
        
        return cka_matrix, layer_names1, layer_names2
    
    def save_results(self, results: Dict, output_path: str):
        """Save CKA analysis results to HDF5 file."""
        print(f"Saving GPU CKA results to: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Save CKA matrices
            if 'inter_model_cka' in results and results['inter_model_cka'] is not None:
                f.create_dataset('inter_model_cka', data=results['inter_model_cka'])
            if 'intra_model1_cka' in results and results['intra_model1_cka'] is not None:
                f.create_dataset('intra_model1_cka', data=results['intra_model1_cka'])
            if 'intra_model2_cka' in results and results['intra_model2_cka'] is not None:
                f.create_dataset('intra_model2_cka', data=results['intra_model2_cka'])
            
            # Save layer names
            f.create_dataset('layer_names1', data=[name.encode() for name in results['layer_names1']])
            if results.get('layer_names2'):
                f.create_dataset('layer_names2', data=[name.encode() for name in results['layer_names2']])
            
            # Save metadata
            f.attrs['kernel'] = self.kernel
            f.attrs['batch_size'] = self.batch_size
            f.attrs['group_size'] = self.group_size
            f.attrs['epsilon'] = self.epsilon
            f.attrs['num_layers1'] = len(results['layer_names1'])
            f.attrs['num_layers2'] = len(results.get('layer_names2', results['layer_names1']))
        
        print("GPU CKA results saved successfully!")
    
    def load_results(self, input_path: str) -> Dict:
        """Load CKA analysis results from HDF5 file."""
        print(f"Loading GPU CKA results from: {input_path}")
        
        with h5py.File(input_path, 'r') as f:
            results = {}
            
            # Load CKA matrices if they exist
            if 'inter_model_cka' in f:
                results['inter_model_cka'] = f['inter_model_cka'][:]
            if 'intra_model1_cka' in f:
                results['intra_model1_cka'] = f['intra_model1_cka'][:]
            if 'intra_model2_cka' in f:
                results['intra_model2_cka'] = f['intra_model2_cka'][:]
            
            # Load layer names
            results['layer_names1'] = [name.decode() for name in f['layer_names1'][:]]
            if 'layer_names2' in f:
                results['layer_names2'] = [name.decode() for name in f['layer_names2'][:]]
            else:
                results['layer_names2'] = results['layer_names1']
            
            # Load metadata
            results['metadata'] = {
                'kernel': f.attrs.get('kernel', 'linear'),
                'batch_size': f.attrs.get('batch_size', 1000),
                'group_size': f.attrs.get('group_size', 64),
                'epsilon': f.attrs.get('epsilon', 1e-4),
                'num_layers1': f.attrs.get('num_layers1', 0),
                'num_layers2': f.attrs.get('num_layers2', 0)
            }
        
        print("GPU CKA results loaded successfully!")
        return results
    
    def plot_cka_matrix(self, cka_matrix: np.ndarray, layer_names1: List[str], 
                       layer_names2: List[str], title: str = "CKA Similarity Matrix",
                       save_path: str = None, figsize: Tuple[int, int] = (12, 10)):
        """Plot CKA similarity matrix as heatmap."""
        plt.figure(figsize=figsize)
        
        # Flip the matrix vertically so diagonal goes from bottom-left to top-right
        cka_matrix_flipped = np.flipud(cka_matrix)
        layer_names1_flipped = layer_names1[::-1]  # Reverse the y-axis labels
        
        # Create heatmap
        sns.heatmap(cka_matrix_flipped, 
                   xticklabels=layer_names2,
                   yticklabels=layer_names1_flipped,
                   annot=True if min(len(layer_names1), len(layer_names2)) <= 10 else False,
                   fmt='.3f',
                   cmap='magma',
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'CKA Similarity'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Model 2 Layers', fontsize=12)
        plt.ylabel('Model 1 Layers', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GPU CKA matrix plot saved to: {save_path}")
        
        plt.show()


class CKAAnalyzer:
    """Class for computing and analyzing CKA similarities between models."""
    
    def __init__(self, kernel: str = 'linear', batch_size: int = 1000, max_samples_for_cka: int = 1500):
        """Initialize CKA analyzer.
        
        Args:
            kernel: Kernel type ('linear' or 'rbf')
            batch_size: Batch size for large datasets
            max_samples_for_cka: Maximum number of samples to use for CKA computation
        """
        self.kernel = kernel
        self.batch_size = batch_size
        self.max_samples_for_cka = max_samples_for_cka
        
        print(f"CKA Analyzer initialized with {kernel} kernel")
        
    def extract_layer_activations(self, model: tf.keras.Model, data: tf.Tensor, 
                                 layer_names: List[str] = None, batch_size: int = 64) -> Dict[str, np.ndarray]:
        """Extract activations from specified layers of a model using batch processing.
        
        Args:
            model: TensorFlow model
            data: Input data tensor
            layer_names: List of layer names to extract from
            batch_size: Batch size for activation extraction to avoid OOM
            
        Returns:
            Dictionary mapping layer names to activation arrays
        """
        print(f"Extracting activations from {len(layer_names)} layers using batch_size={batch_size}")
        
        # Create extraction model
        layer_outputs = []
        valid_layer_names = []
        for layer_name in layer_names:
            try:
                layer = model.get_layer(layer_name)
                layer_outputs.append(layer.output)
                valid_layer_names.append(layer_name)
            except ValueError:
                print(f"Warning: Layer '{layer_name}' not found in model")
                continue
        
        if not layer_outputs:
            raise ValueError("No valid layers found for activation extraction")
        
        extraction_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
        
        # Extract activations in batches
        total_samples = tf.shape(data)[0].numpy()
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        print(f"Processing {total_samples} samples in {num_batches} batches")
        
        result = {}
        first_batch = True
        
        for batch_idx in tqdm(range(num_batches), desc="Extracting activations"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            
            batch_data = data[start_idx:end_idx]
            batch_result = extraction_model(batch_data, training=False)
            if not isinstance(batch_result, list):
                batch_result = [batch_result]
            
            if first_batch:
                for layer_idx, batch_activation in enumerate(batch_result):
                    layer_name = valid_layer_names[layer_idx]
                    if len(batch_activation.shape) > 2:
                        flat_dim = np.prod(batch_activation.shape[1:])
                        shape = (total_samples, flat_dim)
                    else:
                        shape = (total_samples, batch_activation.shape[1])
                    result[layer_name] = np.empty(shape, dtype=np.float32)
                    print(f"Layer '{layer_name}' shape: {shape}")
                first_batch = False
            
            for layer_idx, batch_activation in enumerate(batch_result):
                layer_name = valid_layer_names[layer_idx]
                
                if len(batch_activation.shape) > 2:
                    flattened = tf.reshape(batch_activation, [end_idx - start_idx, -1])
                else:
                    flattened = batch_activation
                
                result[layer_name][start_idx:end_idx] = flattened.numpy()
        
        return result

    def compute_cka_matrix(self, activations1: Dict[str, tf.Tensor], 
                          activations2: Dict[str, tf.Tensor] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """Compute CKA similarity matrix using GPU acceleration when available."""
        # Try to use GPU-accelerated version first
        try:
            gpu_analyzer = GPUCKAAnalyzer(
                kernel=self.kernel,
                batch_size=self.batch_size,
                max_samples_for_cka=self.max_samples_for_cka,
                group_size=min(64, len(list(activations1.keys())))  # Adaptive group size
            )
            return gpu_analyzer.compute_cka_matrix_gpu_batched(activations1, activations2)
        except Exception as e:
            print(f"GPU acceleration failed, falling back to CPU: {e}")
            return self._compute_cka_matrix_cpu_fallback(activations1, activations2)
    
    def _compute_cka_matrix_cpu_fallback(self, activations1: Dict[str, tf.Tensor], 
                                        activations2: Dict[str, tf.Tensor] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """Fallback CPU implementation of CKA matrix computation."""
        is_intra_model = (activations2 is None)
        
        if activations2 is None:
            # Intra-model comparison
            activations2 = activations1
        
        layer_names1 = list(activations1.keys())
        layer_names2 = list(activations2.keys())
        
        # Only intra-model comparison can use symmetric optimization
        # Inter-model comparison must compute full matrix even if square
        is_symmetric = is_intra_model
        
        cka_matrix = np.zeros((len(layer_names1), len(layer_names2)))
        
        print(f"Computing CKA matrix ({len(layer_names1)} x {len(layer_names2)})...")
        if is_intra_model:
            print("Intra-model comparison - computing upper triangle only")
        else:
            print("Inter-model comparison - computing full matrix")
        
        # Calculate total number of computations needed
        total_computations = 0
        for i in range(len(layer_names1)):
            start_j = i if is_symmetric else 0
            total_computations += len(layer_names2) - start_j
        
        print(f"Total CKA computations needed: {total_computations}")
        
        with tqdm(total=total_computations, desc="Computing CKA pairs") as pbar:
            for i, name1 in enumerate(layer_names1):
                start_j = i if is_symmetric else 0
                for j in range(start_j, len(layer_names2)):
                    name2 = layer_names2[j]

                    try:
                        # For intra-model diagonal elements, CKA should be 1.0 (layer with itself)
                        if is_intra_model and i == j:
                            cka_value = 1.0
                        else:
                            act1 = activations1[name1]
                            act2 = activations2[name2]
                            cka_value = cka_numpy(act1, act2, kernel=self.kernel)
                        
                        cka_matrix[i, j] = cka_value
                        
                        if is_symmetric and i != j:
                            cka_matrix[j, i] = cka_matrix[i, j]
                        
                    except Exception as e:
                        print(f"Error computing CKA for {name1} vs {name2}: {e}")
                        cka_matrix[i, j] = 0.0
                        if is_symmetric and i != j:
                            cka_matrix[j, i] = 0.0
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Current': f"{name1[:10]}...vs {name2[:10]}...",
                        'CKA': f"{cka_matrix[i, j]:.3f}" if cka_matrix[i, j] != 0.0 else "Error"
                    })
        
        return cka_matrix, layer_names1, layer_names2
    
    def analyze_models(self, model1: tf.keras.Model, model2: tf.keras.Model, 
                      data: tf.Tensor, layer_names1: List[str] = None, 
                      layer_names2: List[str] = None, extraction_batch_size: int = 64) -> Dict:
        """Comprehensive CKA analysis between two models with GPU optimization.
        
        Args:
            model1: First model
            model2: Second model (None for intra-model only)
            data: Input data for activation extraction
            layer_names1: Layer names for first model
            layer_names2: Layer names for second model
            extraction_batch_size: Batch size for extraction
            
        Returns:
            Dictionary containing analysis results
        """
        # Try to use GPU-accelerated activation extraction
        try:
            gpu_analyzer = GPUCKAAnalyzer(
                kernel=self.kernel,
                batch_size=self.batch_size,
                max_samples_for_cka=self.max_samples_for_cka
            )
            
            # Extract activations from both models using GPU
            print("GPU-accelerated activation extraction for model 1...")
            activations1 = gpu_analyzer.extract_activations_gpu(model1, data, layer_names1)
            
            if model2 is not None:
                print("GPU-accelerated activation extraction for model 2...")
                activations2 = gpu_analyzer.extract_activations_gpu(model2, data, layer_names2)
                
                # Inter-model CKA
                print("Computing inter-model CKA...")
                inter_cka, layer_names1_final, layer_names2_final = self.compute_cka_matrix(activations1, activations2)
                
                # Intra-model CKA for model 2
                print("Computing intra-model CKA for model 2...")
                intra_cka2, _, _ = self.compute_cka_matrix(activations2)
                
            else:
                # Intra-model only
                activations2 = None
                inter_cka = None
                intra_cka2 = None
                layer_names1_final = layer_names1
                layer_names2_final = layer_names1
            
            # Intra-model CKA for model 1
            print("Computing intra-model CKA for model 1...")
            intra_cka1, _, _ = self.compute_cka_matrix(activations1)
            
        except Exception as e:
            print(f"GPU acceleration failed, using CPU fallback: {e}")
            # Fallback to standard CPU extraction
            print("Extracting activations from model 1...")
            activations1 = self.extract_layer_activations(model1, data, layer_names1, extraction_batch_size)
            
            if model2 is not None:
                print("Extracting activations from model 2...")
                activations2 = self.extract_layer_activations(model2, data, layer_names2, extraction_batch_size)
                
                # Inter-model CKA
                print("Computing inter-model CKA...")
                inter_cka, layer_names1_final, layer_names2_final = self.compute_cka_matrix(activations1, activations2)
                
                # Intra-model CKA for model 2
                print("Computing intra-model CKA for model 2...")
                intra_cka2, _, _ = self.compute_cka_matrix(activations2)
            else:
                activations2 = None
                inter_cka = None
                intra_cka2 = None
                layer_names1_final = layer_names1
                layer_names2_final = layer_names1
            
            # Intra-model CKA for model 1
            print("Computing intra-model CKA for model 1...")
            intra_cka1, _, _ = self.compute_cka_matrix(activations1)
        
        results = {
            'inter_model_cka': inter_cka,
            'intra_model1_cka': intra_cka1,
            'intra_model2_cka': intra_cka2,
            'layer_names1': layer_names1_final,
            'layer_names2': layer_names2_final if layer_names2_final else layer_names1_final,
            'activations1': activations1,
            'activations2': activations2
            }
        
        return results
    
    def plot_cka_matrix(self, cka_matrix: np.ndarray, layer_names1: List[str], 
                       layer_names2: List[str], title: str = "CKA Similarity Matrix",
                       save_path: str = None, figsize: Tuple[int, int] = (12, 10)):
        """Plot CKA similarity matrix as heatmap with diagonal from bottom-left to top-right.
        
        Args:
            cka_matrix: CKA similarity matrix
            layer_names1: Layer names for rows
            layer_names2: Layer names for columns
            title: Plot title
            save_path: Path to save the plot
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Flip the matrix vertically so diagonal goes from bottom-left to top-right
        cka_matrix_flipped = np.flipud(cka_matrix)
        layer_names1_flipped = layer_names1[::-1]  # Reverse the y-axis labels
        
        # Create heatmap
        sns.heatmap(cka_matrix_flipped, 
                   xticklabels=layer_names2,
                   yticklabels=layer_names1_flipped,
                   annot=True if min(len(layer_names1), len(layer_names2)) <= 10 else False,
                   fmt='.3f',
                   cmap='magma',
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'CKA Similarity'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Model 2 Layers', fontsize=12)
        plt.ylabel('Model 1 Layers', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CKA matrix plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, output_path: str):
        """Save CKA analysis results to HDF5 file.
        
        Args:
            results: Results dictionary from analyze_models
            output_path: Output file path
        """
        print(f"Saving results to: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Save CKA matrices
            f.create_dataset('inter_model_cka', data=results['inter_model_cka'])
            f.create_dataset('intra_model1_cka', data=results['intra_model1_cka'])
            f.create_dataset('intra_model2_cka', data=results['intra_model2_cka'])
            
            # Save layer names
            f.create_dataset('layer_names1', data=[name.encode() for name in results['layer_names1']])
            f.create_dataset('layer_names2', data=[name.encode() for name in results['layer_names2']])
            
            # Save metadata
            f.attrs['kernel'] = self.kernel
            f.attrs['batch_size'] = self.batch_size
            f.attrs['num_layers1'] = len(results['layer_names1'])
            f.attrs['num_layers2'] = len(results['layer_names2'])
        
        print("Results saved successfully!")
    
    def load_results(self, input_path: str) -> Dict:
        """Load CKA analysis results from HDF5 file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Results dictionary
        """
        print(f"Loading results from: {input_path}")
        
        with h5py.File(input_path, 'r') as f:
            results = {
                'inter_model_cka': f['inter_model_cka'][:],
                'intra_model1_cka': f['intra_model1_cka'][:],
                'intra_model2_cka': f['intra_model2_cka'][:],
                'layer_names1': [name.decode() for name in f['layer_names1'][:]],
                'layer_names2': [name.decode() for name in f['layer_names2'][:]],
                'metadata': {
                    'kernel': f.attrs.get('kernel', 'linear'),
                    'batch_size': f.attrs.get('batch_size', 1000),
                    'num_layers1': f.attrs.get('num_layers1', 0),
                    'num_layers2': f.attrs.get('num_layers2', 0)
                }
            }
        
        print("Results loaded successfully!")
        return results


def summarize_cka_analysis(results: Dict) -> Dict:
    """Generate summary statistics from CKA analysis results.
    
    Args:
        results: Results dictionary from CKAAnalyzer.analyze_models
        
    Returns:
        Summary statistics dictionary
    """
    inter_cka = results['inter_model_cka']
    intra_cka1 = results['intra_model1_cka']
    intra_cka2 = results['intra_model2_cka']
    
    # Mask diagonal for intra-model analysis (self-similarity = 1)
    intra_cka1_no_diag = intra_cka1[~np.eye(intra_cka1.shape[0], dtype=bool)]
    intra_cka2_no_diag = intra_cka2[~np.eye(intra_cka2.shape[0], dtype=bool)]
    
    summary = {
        'inter_model': {
            'mean': float(np.mean(inter_cka)),
            'std': float(np.std(inter_cka)),
            'max': float(np.max(inter_cka)),
            'min': float(np.min(inter_cka)),
            'median': float(np.median(inter_cka))
        },
        'intra_model1': {
            'mean': float(np.mean(intra_cka1_no_diag)),
            'std': float(np.std(intra_cka1_no_diag)),
            'max': float(np.max(intra_cka1_no_diag)),
            'min': float(np.min(intra_cka1_no_diag)),
            'median': float(np.median(intra_cka1_no_diag))
        },
        'intra_model2': {
            'mean': float(np.mean(intra_cka2_no_diag)),
            'std': float(np.std(intra_cka2_no_diag)),
            'max': float(np.max(intra_cka2_no_diag)),
            'min': float(np.min(intra_cka2_no_diag)),
            'median': float(np.median(intra_cka2_no_diag))
        }
    }
    
    return summary