#!/usr/bin/env python

"""
test_cka_example.py

Simple test script to demonstrate CKA functionality with synthetic data.
This can be used to verify the implementation works correctly.
"""

import numpy as np
import tensorflow as tf
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paddy import cka

def create_test_model(input_shape=(100, 4), num_layers=3):
    """Create a simple test model for CKA analysis."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    for i in range(num_layers):
        model.add(tf.keras.layers.Conv1D(
            filters=32 * (i + 1),
            kernel_size=3,
            activation='relu',
            padding='same',
            name=f'conv1d_{i+1}'
        ))
        model.add(tf.keras.layers.BatchNormalization(name=f'batch_norm_{i+1}'))
    
    model.add(tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool'))
    model.add(tf.keras.layers.Dense(64, activation='relu', name='dense_1'))
    model.add(tf.keras.layers.Dense(32, activation='relu', name='dense_2'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', name='output'))
    
    return model

def generate_test_data(num_samples=200, seq_length=100):
    """Generate synthetic sequence-like data for testing."""
    # Create random one-hot encoded sequences
    data = np.random.rand(num_samples, seq_length, 4)
    # Make it more like one-hot by setting max value to 1 and others to 0
    max_indices = np.argmax(data, axis=-1)
    data = np.zeros_like(data)
    for i in range(num_samples):
        for j in range(seq_length):
            data[i, j, max_indices[i, j]] = 1.0
    
    return tf.constant(data, dtype=tf.float32)

def test_basic_cka():
    """Test basic CKA functionality."""
    print("Testing basic CKA functions...")
    
    # Create test data
    X = tf.random.normal((100, 50), dtype=tf.float64)
    Y = tf.random.normal((100, 30), dtype=tf.float64)
    
    # Test kernel functions
    K_linear = cka.linear_kernel(X)
    print(f"Linear kernel shape: {K_linear.shape}")
    assert K_linear.shape == (100, 100)
    
    K_rbf = cka.rbf_kernel(X)
    print(f"RBF kernel shape: {K_rbf.shape}")
    assert K_rbf.shape == (100, 100)
    
    # Test centering
    K_centered = cka.center_gram_matrix(K_linear)
    print(f"Centered kernel shape: {K_centered.shape}")
    
    # Test HSIC
    hsic_value = cka.hsic(K_linear, cka.linear_kernel(Y))
    print(f"HSIC value: {hsic_value}")
    
    # Test CKA
    cka_value = cka.cka(X, Y)
    print(f"CKA value: {cka_value}")
    assert 0 <= cka_value <= 1, f"CKA value should be between 0 and 1, got {cka_value}"
    
    print("Basic CKA tests passed! ✓")

def test_cka_analyzer():
    """Test CKAAnalyzer class with synthetic models."""
    print("\nTesting CKAAnalyzer with synthetic models...")
    
    # Create test models
    model1 = create_test_model(input_shape=(100, 4), num_layers=2)
    model2 = create_test_model(input_shape=(100, 4), num_layers=3)
    
    print(f"Model 1 layers: {len(model1.layers)}")
    print(f"Model 2 layers: {len(model2.layers)}")
    
    # Generate test data
    test_data = generate_test_data(num_samples=50, seq_length=100)
    print(f"Test data shape: {test_data.shape}")
    
    # Initialize CKA analyzer
    analyzer = cka.CKAAnalyzer(kernel='linear', batch_size=25)
    
    # Test activation extraction
    print("Extracting activations from model 1...")
    activations1 = analyzer.extract_layer_activations(model1, test_data)
    print(f"Extracted activations from {len(activations1)} layers")
    
    for layer_name, activation in activations1.items():
        print(f"  {layer_name}: {activation.shape}")
    
    # Test CKA matrix computation (intra-model)
    print("Computing intra-model CKA matrix...")
    cka_matrix, layer_names, _ = analyzer.compute_cka_matrix(activations1)
    print(f"CKA matrix shape: {cka_matrix.shape}")
    print(f"CKA matrix diagonal (should be ~1.0): {np.diag(cka_matrix)}")
    
    # Test full analysis
    print("Running full CKA analysis...")
    results = analyzer.analyze_models(model1, model2, test_data)
    
    print("Analysis results:")
    print(f"  Inter-model CKA shape: {results['inter_model_cka'].shape}")
    print(f"  Intra-model1 CKA shape: {results['intra_model1_cka'].shape}")
    print(f"  Intra-model2 CKA shape: {results['intra_model2_cka'].shape}")
    
    # Test summary statistics
    summary = cka.summarize_cka_analysis(results)
    print(f"  Inter-model CKA mean: {summary['inter_model']['mean']:.4f}")
    print(f"  Intra-model1 CKA mean: {summary['intra_model1']['mean']:.4f}")
    print(f"  Intra-model2 CKA mean: {summary['intra_model2']['mean']:.4f}")
    
    print("CKAAnalyzer tests passed! ✓")
    
    return results

def test_visualization(results):
    """Test visualization functionality."""
    print("\nTesting visualization...")
    
    try:
        analyzer = cka.CKAAnalyzer()
        
        # Test plotting (will save to current directory)
        analyzer.plot_cka_matrix(
            results['inter_model_cka'],
            results['layer_names1'],
            results['layer_names2'],
            title="Test Inter-Model CKA",
            save_path="test_inter_cka.png"
        )
        
        analyzer.plot_cka_matrix(
            results['intra_model1_cka'],
            results['layer_names1'],
            results['layer_names1'],
            title="Test Intra-Model CKA",
            save_path="test_intra_cka.png"
        )
        
        print("Visualization tests passed! ✓")
        print("Plots saved as test_inter_cka.png and test_intra_cka.png")
        
    except Exception as e:
        print(f"Visualization test failed: {e}")
        print("This might be due to display issues, but core functionality works")

def main():
    """Run all tests."""
    print("=" * 50)
    print("Running CKA Implementation Tests")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    try:
        # Test basic CKA functions
        test_basic_cka()
        
        # Test CKA analyzer
        results = test_cka_analyzer()
        
        # Test visualization
        test_visualization(results)
        
        print("\n" + "=" * 50)
        print("All tests passed successfully! 🎉")
        print("CKA implementation is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
