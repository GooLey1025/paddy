#!/usr/bin/env python

"""
cka_repr_models_compute.py

Compute CKA similarities between neural network models using TFRecord data.
Supports both inter-model and intra-model CKA analysis.

"""

import argparse
import os
import sys
import yaml
import numpy as np
from typing import Dict, List, Optional, Tuple
import h5py
from tqdm import tqdm
import json
import time

from paddy import seqnn
from paddy import dataset
from paddy import utils
from paddy import cka
# Set GPU device(s) before importing TensorFlow
gpu_ids = utils.set_gpu_device()
import tensorflow as tf


class ModelCKAComputer:
    
    def __init__(self, kernel: str = 'linear', batch_size: int = 32, max_samples: int = None, 
                 use_gpu_acceleration: bool = True, group_size: int = 64):
        """Initialize the GPU-accelerated CKA computer.
        
        Args:
            kernel: Kernel type for CKA computation ('linear' or 'rbf')
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to process (None = use all)
            use_gpu_acceleration: Whether to use GPU acceleration
            group_size: Group size for batched GPU operations
        """
        self.kernel = kernel
        self.batch_size = batch_size
        self.use_gpu_acceleration = use_gpu_acceleration
        self.group_size = group_size
        
        # Round up max_samples to batch_size multiple if specified
        if max_samples is not None:
            max_batches = (max_samples + batch_size - 1) // batch_size
            self.max_samples = max_batches * batch_size
            self.max_batches = max_batches
            print(f"Requested {max_samples} samples, rounded up to {self.max_samples} ({max_batches} batches)")
        else:
            self.max_samples = None
            self.max_batches = None

        # Initialize CKA analyzer with GPU acceleration
        if use_gpu_acceleration:
            self.cka_analyzer = cka.GPUCKAAnalyzer(
                kernel=kernel, 
                batch_size=batch_size, 
                max_samples_for_cka=max_samples,
                group_size=group_size
            )
            print("Using GPU-accelerated CKA analyzer")
        else:
            self.cka_analyzer = cka.CKAAnalyzer(
                kernel=kernel, 
                batch_size=batch_size, 
                max_samples_for_cka=max_samples
            )
            print("Using standard CKA analyzer")
        
        print(f"Initialized CKA Computer:")
        print(f"  - Kernel: {kernel}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Max samples: {self.max_samples}")
        print(f"  - GPU acceleration: {use_gpu_acceleration}")
        if use_gpu_acceleration:
            print(f"  - Group size: {group_size}")
    
    def load_model(self, params_file: str, weights_file: str) -> tf.keras.Model:
        """Load a model from parameter file and weights.
        
        Args:
            params_file: Path to YAML parameter file
            weights_file: Path to model weights file
            
        Returns:
            Loaded TensorFlow model
        """
        print(f"Loading model from {params_file} and {weights_file}")
        
        # Load parameters
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
        
        params_model = params["model"]
        self.model_type = params_model["model_type"]
        params_model["verbose"] = False


        model = seqnn.SeqNN(params_model)
        
        # Load weights
        model.restore(weights_file, trunk=False)
        
        print(f"Successfully loaded model")
        print(f"Model input shape: {model.model.input_shape}")
        print(f"Model output shape: {model.model.output_shape}")
        
        return model.model
    
    def load_data_from_tfrecords(self, data_dir: str, split: str = 'test') -> tf.data.Dataset:
        """Load data from TFRecord dataset.
        
        Args:
            data_dir: Path to TFRecord data directory
            split: Dataset split to use ('train', 'valid', 'test')
            
        Returns:
            TensorFlow dataset
        """
        print(f"Loading data from TFRecord: {data_dir}")
        print(f"Using split: {split}")
        
        # Create dataset
        seq_data = dataset.SeqDataset(
            data_dir,
            split_label=split,
            batch_size=self.batch_size,
            mode="eval",
            model_type=self.model_type,
        )
        
        # Get the dataset
        data_dataset = seq_data.dataset
        
        # Limit samples if specified
        if self.max_batches is not None:
            data_dataset = data_dataset.take(self.max_batches)
            print(f"Limited to {self.max_batches} batches ({self.max_samples} samples)")
        
        return data_dataset
    
    def select_meaningful_layers(self, model: tf.keras.Model) -> List[str]:
        """Select meaningful layers for representation extraction.
        
        Args:
            model: TensorFlow model
            
        Returns:
            List of selected layer names
        """
        selected = []
        skip_types = ['input', 'dropout', 'reshape', 'flatten', 'lambda', 'concatenate']
        
        print("=== Available Layers ===")
        print(f"{'Index':<6} {'Layer Name':<40} {'Layer Type':<25} {'Output Shape':<20}")
        print("-" * 95)
        
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__.lower()
            
            try:
                output_shape = str(layer.output_shape)
            except:
                output_shape = "Unknown"
            
            print(f"{i:<6} {layer.name:<40} {type(layer).__name__:<25} {output_shape:<20}")
            
            # Skip layers that don't have meaningful representations
            if any(skip_type in layer_type for skip_type in skip_types):
                continue
            
            # Include layers with activations or major computation
            if (hasattr(layer, 'activation') or 
                any(keyword in layer_type for keyword in ['conv', 'dense', 'attention', 'norm', 'batch'])):
                selected.append(layer.name)
        
        print(f"Selected {len(selected)} meaningful layers for extraction:")
        for layer_name in selected:
            print(f"  - {layer_name}")
        
        return selected
    
    def _extract_sequences_from_batch(self, batch_data):
        """Extract sequences from batch data (handle both tuple and tensor formats)."""
        if isinstance(batch_data, tuple) and len(batch_data) >= 2:
            return batch_data[0]
        else:
            return batch_data
    
    def compute_model_cka(self, model1: tf.keras.Model, model2: tf.keras.Model,
                         data_dataset: tf.data.Dataset, layer_names1: List[str], 
                         layer_names2: List[str] = None, intra_only: bool = False) -> Dict:
        """Compute CKA similarities between models with GPU optimization.
        
        Args:
            model1: First model
            model2: Second model (can be None for intra-model analysis)
            data_dataset: Input dataset
            layer_names1: Layer names for first model
            layer_names2: Layer names for second model (None = use layer_names1)
            intra_only: Only compute intra-model CKA for model1
            
        Returns:
            Dictionary containing CKA analysis results
        """
        # Convert dataset to tensor with GPU memory optimization
        print("Converting dataset to tensor with GPU optimization...")
        data_list = []
        total_samples = 0
        
        # Use GPU memory more efficiently
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            for batch_data in tqdm(data_dataset, desc="Loading data"):
                batch_sequences = self._extract_sequences_from_batch(batch_data)
                # Keep data on GPU if possible
                if tf.config.list_physical_devices('GPU'):
                    data_list.append(batch_sequences)
                else:
                    data_list.append(batch_sequences.numpy())
                total_samples += tf.shape(batch_sequences)[0].numpy()
            
            # Concatenate all batches efficiently
            if tf.config.list_physical_devices('GPU'):
                data_tensor = tf.concat(data_list, axis=0)
            else:
                data_tensor = tf.concat([tf.constant(batch) for batch in data_list], axis=0)
            
        print(f"Loaded {total_samples} samples for GPU-accelerated CKA analysis")
        
        # Add memory management
        def clear_gpu_memory():
            if tf.config.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
        
        if intra_only:
            # Only compute intra-model CKA for model1
            print("Computing GPU-accelerated intra-model CKA analysis...")
            
            if self.use_gpu_acceleration:
                # Use GPU-accelerated extraction
                activations1 = self.cka_analyzer.extract_activations_gpu(model1, data_tensor, layer_names1)
                intra_cka1, layer_names1_final, _ = self.cka_analyzer.compute_cka_matrix_gpu_batched(activations1)
                
                results = {
                    'intra_model1_cka': intra_cka1,
                    'layer_names1': layer_names1_final,
                    'layer_names2': layer_names1_final,
                    'activations1': activations1,
                    'activations2': activations1
                }
            else:
                # Fallback to standard analyzer
                results = self.cka_analyzer.analyze_models(
                    model1, None, data_tensor, layer_names1, layer_names1,
                    self.batch_size
                )
            
            results['analysis_type'] = 'intra_model'
        else:   
            if layer_names2 is None:
                layer_names2 = layer_names1
            
            print("Computing GPU-accelerated inter-model and intra-model CKA analysis...")
            
            if self.use_gpu_acceleration:
                # Use GPU-accelerated extraction
                print("GPU-accelerated activation extraction for model 1...")
                activations1 = self.cka_analyzer.extract_activations_gpu(model1, data_tensor, layer_names1)
                
                print("GPU-accelerated activation extraction for model 2...")
                activations2 = self.cka_analyzer.extract_activations_gpu(model2, data_tensor, layer_names2)
                
                # Compute CKA matrices using GPU acceleration
                print("Computing inter-model CKA with GPU acceleration...")
                inter_cka, layer_names1_final, layer_names2_final = self.cka_analyzer.compute_cka_matrix_gpu_batched(activations1, activations2)
                
                print("Computing intra-model CKA for model 1...")
                intra_cka1, _, _ = self.cka_analyzer.compute_cka_matrix_gpu_batched(activations1)
                
                print("Computing intra-model CKA for model 2...")
                intra_cka2, _, _ = self.cka_analyzer.compute_cka_matrix_gpu_batched(activations2)
                
                results = {
                    'inter_model_cka': inter_cka,
                    'intra_model1_cka': intra_cka1,
                    'intra_model2_cka': intra_cka2,
                    'layer_names1': layer_names1_final,
                    'layer_names2': layer_names2_final,
                    'activations1': activations1,
                    'activations2': activations2
                }
            else:
                # Fallback to standard analyzer
                results = self.cka_analyzer.analyze_models(
                    model1, model2, data_tensor, layer_names1, layer_names2,
                    self.batch_size
                )
            
            results['analysis_type'] = 'inter_intra_model'
        
        # Clear GPU memory after computation
        clear_gpu_memory()
        
        return results
    
    def generate_report(self, results: Dict, output_dir: str):
        """Generate comprehensive CKA analysis report.
        
        Args:
            results: CKA analysis results
            output_dir: Output directory for report
        """
        report_path = os.path.join(output_dir, "cka_analysis_report.txt")
        
        # Generate summary statistics
        summary = cka.summarize_cka_analysis(results)
        
        print(f"Generating analysis report: {report_path}")
        
        with open(report_path, 'w') as f:
            f.write("CKA Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Type: {results.get('analysis_type', 'unknown')}\n")
            f.write(f"Kernel: {self.kernel}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Max Samples: {self.max_samples}\n\n")
            
            # Model information
            f.write("Model Information:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Model 1 Layers: {len(results['layer_names1'])}\n")
            if 'layer_names2' in results and results['layer_names2'] != results['layer_names1']:
                f.write(f"Model 2 Layers: {len(results['layer_names2'])}\n")
            f.write("\n")
            
            # Summary statistics
            if 'inter_model_cka' in results:
                f.write("Inter-Model CKA Statistics:\n")
                f.write("-" * 30 + "\n")
                inter_stats = summary['inter_model']
                f.write(f"Mean: {inter_stats['mean']:.4f}\n")
                f.write(f"Std:  {inter_stats['std']:.4f}\n")
                f.write(f"Max:  {inter_stats['max']:.4f}\n")
                f.write(f"Min:  {inter_stats['min']:.4f}\n")
                f.write(f"Median: {inter_stats['median']:.4f}\n\n")
            
            if 'intra_model1_cka' in results:
                f.write("Intra-Model 1 CKA Statistics:\n")
                f.write("-" * 30 + "\n")
                intra_stats = summary['intra_model1']
                f.write(f"Mean: {intra_stats['mean']:.4f}\n")
                f.write(f"Std:  {intra_stats['std']:.4f}\n")
                f.write(f"Max:  {intra_stats['max']:.4f}\n")
                f.write(f"Min:  {intra_stats['min']:.4f}\n")
                f.write(f"Median: {intra_stats['median']:.4f}\n\n")
                
            if 'intra_model2_cka' in results:
                f.write("Intra-Model 2 CKA Statistics:\n")
                f.write("-" * 30 + "\n")
                intra_stats = summary['intra_model2']
                f.write(f"Mean: {intra_stats['mean']:.4f}\n")
                f.write(f"Std:  {intra_stats['std']:.4f}\n")
                f.write(f"Max:  {intra_stats['max']:.4f}\n")
                f.write(f"Min:  {intra_stats['min']:.4f}\n")
                f.write(f"Median: {intra_stats['median']:.4f}\n\n")
            
            # Layer information
            f.write("Layer Information:\n")
            f.write("-" * 20 + "\n")
            f.write("Model 1 Layers:\n")
            for i, layer_name in enumerate(results['layer_names1']):
                f.write(f"  {i:2d}: {layer_name}\n")
            
            if 'layer_names2' in results and results['layer_names2'] != results['layer_names1']:
                f.write("\nModel 2 Layers:\n")
                for i, layer_name in enumerate(results['layer_names2']):
                    f.write(f"  {i:2d}: {layer_name}\n")
        
        # Save summary as JSON for programmatic access
        summary_path = os.path.join(output_dir, "cka_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Report saved to: {report_path}")
        print(f"Summary saved to: {summary_path}")
    
    def generate_plots(self, results: Dict, output_dir: str):
        """Generate visualization plots for CKA analysis.
        
        Args:
            results: CKA analysis results
            output_dir: Output directory for plots
        """
        print("Generating visualization plots...")
        
        # Inter-model CKA plot
        if 'inter_model_cka' in results:
            inter_path = os.path.join(output_dir, "inter_model_cka.png")
            self.cka_analyzer.plot_cka_matrix(
                results['inter_model_cka'],
                results['layer_names1'],
                results['layer_names2'],
                title="Inter-Model CKA Similarity",
                save_path=inter_path,
                figsize=(14, 12)
            )
        
        # Intra-model CKA plots
        if 'intra_model1_cka' in results:
            intra1_path = os.path.join(output_dir, "intra_model1_cka.png")
            self.cka_analyzer.plot_cka_matrix(
                results['intra_model1_cka'],
                results['layer_names1'],
                results['layer_names1'],
                title="Intra-Model 1 CKA Similarity",
                save_path=intra1_path,
                figsize=(12, 10)
            )
        
        if 'intra_model2_cka' in results:
            intra2_path = os.path.join(output_dir, "intra_model2_cka.png")
            self.cka_analyzer.plot_cka_matrix(
                results['intra_model2_cka'],
                results['layer_names2'],
                results['layer_names2'],
                title="Intra-Model 2 CKA Similarity",
                save_path=intra2_path,
                figsize=(12, 10)
            )
        
        print("Plots generated successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Compute CKA similarities between neural network models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models
  cka_repr_models_compute.py \\
    --model1_params model1.yaml --model1_weights model1.h5 \\
    --model2_params model2.yaml --model2_weights model2.h5 \\
    --data_dir /path/to/tfrecords --output_dir cka_results

  # Analyze single model (intra-model only)
  cka_repr_models_compute.py \\
    --model1_params model.yaml --model1_weights model.h5 \\
    --data_dir /path/to/tfrecords --output_dir cka_results --intra_only

  # Generate plots only from existing results
  cka_repr_models_compute.py \\
    --output_dir cka_results --plot_only
        """
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("-m1p","--model1_params",
                           help="YAML parameter file for first model (required unless --plot_only)")
    model_group.add_argument("-m1w","--model1_weights",
                           help="Weights file for first model (required unless --plot_only)")
    model_group.add_argument("-m2p","--model2_params",
                           help="YAML parameter file for second model")
    model_group.add_argument("-m2w","--model2_weights",
                           help="Weights file for second model")
    
    # Data arguments
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("-d","--data_dir",
                          help="TFRecord data directory (required unless --plot_only)")
    data_group.add_argument("--split", default="test", choices=['train', 'valid', 'test'],
                          help="Dataset split to use (default: test)")
    
    # Layer selection arguments
    layer_group = parser.add_argument_group("Layer Selection")
    layer_group.add_argument("-m1l","--model1_layers", nargs='+',
                           help="Specific layer names for model 1")
    layer_group.add_argument("-m2l","--model2_layers", nargs='+',
                           help="Specific layer names for model 2")
    
    # Analysis arguments
    analysis_group = parser.add_argument_group("Analysis Configuration")
    analysis_group.add_argument("-k","--kernel", choices=['linear', 'rbf'], default='linear',
                              help="Kernel type for CKA computation")
    analysis_group.add_argument("-bs","--batch_size", type=int, default=32,
                              help="Batch size for processing (default: 32)")
    analysis_group.add_argument("-ms","--max_samples", type=int, default=None,
                              help="Maximum number of samples to process (default: all). It will be round up to the nearest multiple of the batch size. Strongly recommend to use a smaller number because its too memory intensive.")
    analysis_group.add_argument("-io","--intra_only", action='store_true',
                              help="Only perform intra-model analysis for model1")
    analysis_group.add_argument("--no_gpu", action='store_true',
                              help="Disable GPU acceleration and use CPU-only computation")
    analysis_group.add_argument("-gs","--group_size", type=int, default=64,
                              help="Group size for batched GPU operations (default: 64, inspired by PyTorch implementation)")

    
    # Output arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("-o","--output_dir", default="cka_output",
                            help="Output directory for results")
    output_group.add_argument("-po","--plot_only", action='store_true',
                            help="Only generate plots and reports from existing cka_results.h5 file in output_dir (skips model loading and computation)")
    args = parser.parse_args()
    
    # Validation: model parameters are required unless in plot_only mode
    if not args.plot_only:
        # Check required parameters for model analysis
        if not args.model1_params or not args.model1_weights:
            parser.error("model1_params and model1_weights are required unless --plot_only is specified")
        
        if not args.data_dir:
            parser.error("data_dir is required unless --plot_only is specified")
        
        if not args.intra_only and (not args.model2_params or not args.model2_weights):
            parser.error("model2_params and model2_weights required unless --intra_only is specified")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    
    try:
        if args.plot_only:
            results_path = os.path.join(args.output_dir, "cka_results.h5")
            if os.path.exists(results_path):
                print("plot_only mode: Loading existing CKA results and generating plots...")
                
                # Initialize CKA analyzer to load results and generate plots
                cka_computer = ModelCKAComputer(
                    kernel=args.kernel,
                    batch_size=args.batch_size,
                    max_samples=args.max_samples,
                    use_gpu_acceleration=not args.no_gpu,
                    group_size=args.group_size
                )
                
                # Load existing results
                print(f"Loading results from: {results_path}")
                results = cka_computer.cka_analyzer.load_results(results_path)
                
                # Generate report
                print("Generating analysis report...")
                cka_computer.generate_report(results, args.output_dir)
                
                # Generate plots
                print("Generating visualization plots...")
                cka_computer.generate_plots(results, args.output_dir)
                
                print(f"\n*** Plot-only mode complete! ***")
                print(f"*** Results loaded from: {results_path} ***")
                print(f"*** New plots and reports saved to: {args.output_dir} ***")
                
                # Print quick summary
                if 'inter_model_cka' in results:
                    inter_mean = np.mean(results['inter_model_cka'])
                    print(f"*** Inter-model CKA mean similarity: {inter_mean:.4f} ***")
                
                if 'intra_model1_cka' in results:
                    intra_mean1 = np.mean(results['intra_model1_cka'][np.triu_indices_from(results['intra_model1_cka'], k=1)])
                    print(f"*** Intra-model 1 CKA mean similarity: {intra_mean1:.4f} ***")
                
                if 'intra_model2_cka' in results:
                    intra_mean2 = np.mean(results['intra_model2_cka'][np.triu_indices_from(results['intra_model2_cka'], k=1)])
                    print(f"*** Intra-model 2 CKA mean similarity: {intra_mean2:.4f} ***")
                    
            else:
                print(f"Error: cka_results.h5 not found in {args.output_dir}")
                print("Please run the analysis first without --plot_only flag")
                return 1
        else:
            # Initialize CKA computer with GPU acceleration
            cka_computer = ModelCKAComputer(
                kernel=args.kernel,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                use_gpu_acceleration=not args.no_gpu,
                group_size=args.group_size
            )
            
            # Load models
            print("Loading model 1...")
            model1 = cka_computer.load_model(args.model1_params, args.model1_weights)
            
            model2 = None
            if not args.intra_only:
                print("Loading model 2...")
                model2 = cka_computer.load_model(args.model2_params, args.model2_weights)
            
            # Load data
            print("Loading data...")
            data_dataset = cka_computer.load_data_from_tfrecords(args.data_dir, args.split)
            
            # Select layers
            if args.model1_layers:
                layer_names1 = args.model1_layers
                print(f"Using specified layers for model 1: {layer_names1}")
            else:
                print("\n\nSelecting meaningful layers for model 1...")
                layer_names1 = cka_computer.select_meaningful_layers(model1)
            
            layer_names2 = None
            if model2 is not None:
                if args.model2_layers:
                    layer_names2 = args.model2_layers
                    print(f"Using specified layers for model 2: {layer_names2}")
                else:
                    print("\n\nSelecting meaningful layers for model 2...")
                    layer_names2 = cka_computer.select_meaningful_layers(model2)
            
            # Compute CKA analysis
            print("\n\nComputing CKA analysis...")
            results = cka_computer.compute_model_cka(
                model1, model2, data_dataset, layer_names1, layer_names2, args.intra_only
            )
            
            # Save results
            print("Saving results...")
            results_path = os.path.join(args.output_dir, "cka_results.h5")
            cka_computer.cka_analyzer.save_results(results, results_path)
            
            
            cka_computer.generate_report(results, args.output_dir)
            
            cka_computer.generate_plots(results, args.output_dir)
            
            print(f"\n*** CKA Analysis Complete! ***")
            print(f"*** Results saved to: {args.output_dir} ***")
            print(f"*** Main results file: {results_path} ***")
            
            # Print quick summary
            if 'inter_model_cka' in results:
                inter_mean = np.mean(results['inter_model_cka'])
                print(f"*** Inter-model CKA mean similarity: {inter_mean:.4f} ***")
            
            if 'intra_model1_cka' in results:
                intra_mean1 = np.mean(results['intra_model1_cka'][np.triu_indices_from(results['intra_model1_cka'], k=1)])
                print(f"*** Intra-model 1 CKA mean similarity: {intra_mean1:.4f} ***")
            
            if 'intra_model2_cka' in results:
                intra_mean2 = np.mean(results['intra_model2_cka'][np.triu_indices_from(results['intra_model2_cka'], k=1)])
                print(f"*** Intra-model 2 CKA mean similarity: {intra_mean2:.4f} ***")
        
    except Exception as e:
        print(f"Error during CKA analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # debug mode with GPU acceleration testing
    import sys
    if len(sys.argv) == 1:
        sys.argv += [
            "-m1p", "/data/xhhuanglab/gulei/projects/paddy/cka/23ave.params_micro_106targets.yaml",
            "-m1w", "/data/xhhuanglab/gulei/projects/paddy/seq2exp/SC_models/34P_Borzoi_Seq2RNAseqTracks_rp1_model_best.h5",
            "-m2p", "/data/xhhuanglab/gulei/projects/paddy/cka/23ave.params_micro_106targets.yaml",
            "-m2w", "/data/xhhuanglab/gulei/projects/paddy/seq2exp/SC_models/34P_Borzoi_Seq2RNAseqTracks_rp2_model_best.h5",
            "-d", "/data/xhhuanglab/gulei/projects/paddy/seq2exp/23_P8_mseqs",
            "-ms", "50",
            "--group_size", "32",  # Smaller group size for testing
            "-o", "debug_gpu"
        ]
    
    # Print GPU information
    print("=== GPU Information ===")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        # Enable memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPUs found, using CPU")
    
    main()
