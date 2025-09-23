#!/usr/bin/env python

"""
models_cka_compute.py

Compute Centered Kernel Alignment (CKA) similarities between neural network models.
This script can analyze both inter-model similarities (between different models) and 
intra-model similarities (within a single model's layers).

Features:
- Load models with YAML parameter files
- Extract layer activations using representative data
- Compute CKA similarity matrices
- Generate visualization plots
- Save results for further analysis

Usage Examples:
    # Compare two models
    python models_cka_compute.py \\
        --model1_params model1_params.yaml --model1_weights model1.h5 \\
        --model2_params model2_params.yaml --model2_weights model2.h5 \\
        --data_source bed --bed_file sequences.bed --genome_fasta genome.fa \\
        --output_dir cka_results

    # Analyze single model (intra-model)
    python models_cka_compute.py \\
        --model1_params model_params.yaml --model1_weights model.h5 \\
        --data_source sequences --sequences ATCGATCG... GCTAGCTA... \\
        --output_dir cka_results --intra_only

Author: Assistant
"""

import argparse
import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import h5py

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paddy import seqnn
from paddy import dataset
from paddy import bed
from paddy import dna
from paddy import utils
from paddy import cka


class ModelCKAComputer:
    """Main class for computing CKA similarities between models."""
    
    def __init__(self, kernel: str = 'linear', batch_size: int = 500, 
                 max_samples: int = None, random_seed: int = 42, 
                 extraction_batch_size: int = None, use_memmap: bool = False,
                 use_fast_extraction: bool = True):
        """Initialize the CKA computer.
        
        Args:
            kernel: Kernel type for CKA computation ('linear' or 'rbf')
            batch_size: Batch size for processing
            max_samples: Maximum number of samples to use for CKA analysis (None = use all)
            random_seed: Random seed for reproducibility
            extraction_batch_size: Batch size for activation extraction (None = auto-optimize)
            use_memmap: Use memory mapping for activations to reduce RAM usage
        """
        self.kernel = kernel
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.extraction_batch_size = extraction_batch_size
        self.use_memmap = use_memmap
        self.use_fast_extraction = use_fast_extraction
        
        # Set random seeds
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize CKA analyzer
        self.cka_analyzer = cka.CKAAnalyzer(kernel=kernel, batch_size=batch_size, max_samples_for_cka=max_samples)
        
        print(f"Initialized CKA computer:")
        print(f"  - Kernel: {kernel}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Extraction batch size: {extraction_batch_size if extraction_batch_size else 'auto-optimize'}")
        print(f"  - Use memory mapping: {use_memmap}")
        print(f"  - Use fast extraction: {use_fast_extraction}")
        print(f"  - Max samples for analysis: {max_samples}")
        print(f"  - Random seed: {random_seed}")
    
    def load_model(self, params_file: str, weights_file: str, model_name: str = "model") -> tf.keras.Model:
        """Load a model from parameter file and weights.
        
        Args:
            params_file: Path to YAML parameter file
            weights_file: Path to model weights file
            model_name: Name for the model (for logging)
            
        Returns:
            Loaded TensorFlow model
        """
        print(f"Loading {model_name} from {params_file} and {weights_file}")
        
        # Load parameters
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
        
        params_model = params["model"]  # Handle both nested and flat structure
        self.model_type = params_model["model_type"]
        params_model["verbose"] = False  # Reduce verbosity during loading

        # Create model
        if "model_type" in params_model and params_model["model_type"] == "slope":
            # Handle SlopeNN models
            model = seqnn.SlopeNN(params_model)
        else:
            # Handle regular SeqNN models
            model = seqnn.SeqNN(params_model)
        
        # Load weights
        model.restore(weights_file, trunk=False)
        
        print(f"Successfully loaded {model_name}")
        print(f"Model input shape: {model.model.input_shape}")
        print(f"Model output shape: {model.model.output_shape}")
        
        return model.model
    
    def load_data(self, data_source: str, **kwargs) -> tf.Tensor:
        """Load and prepare data for CKA computation.
        
        Args:
            data_source: Type of data source ('bed', 'fasta', 'sequences', 'tfrecord')
            **kwargs: Additional arguments specific to data source
            
        Returns:
            Data tensor ready for model inference
        """
        print(f"Loading data from source: {data_source}")
        
        sequences = []
        
        if data_source == 'bed':
            bed_file = kwargs.get('bed_file')
            genome_fasta = kwargs.get('genome_fasta')
            seq_length = kwargs.get('seq_length', 32768)
            
            if not bed_file or not genome_fasta:
                raise ValueError("bed_file and genome_fasta required for BED data source")
            
            print(f"Loading sequences from BED file: {bed_file}")
            model_sequences, model_coords = bed.make_bed_seqs(
                bed_file, genome_fasta, seq_length, stranded=False
            )
            sequences = model_sequences
            
        elif data_source == 'fasta':
            fasta_file = kwargs.get('fasta_file')
            if not fasta_file:
                raise ValueError("fasta_file required for FASTA data source")
            
            print(f"Loading sequences from FASTA file: {fasta_file}")
            with open(fasta_file, 'r') as f:
                seq = ""
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if seq:
                            sequences.append(seq.upper())
                            seq = ""
                    else:
                        seq += line
                if seq:
                    sequences.append(seq.upper())
                    
        elif data_source == 'sequences':
            direct_sequences = kwargs.get('sequences', [])
            if not direct_sequences:
                raise ValueError("sequences required for direct sequence input")
            sequences = [seq.upper() for seq in direct_sequences]
            
        elif data_source == 'tfrecord':
            # Handle TFRecord data (for compatibility with existing datasets)
            data_dir = kwargs.get('data_dir')
            split = kwargs.get('split', 'test')
            batch_size = kwargs.get('batch_size', self.batch_size)
                        
            if not data_dir:
                raise ValueError("data_dir required for TFRecord data source")
            
            print(f"Loading data from TFRecord: {data_dir}")
            seq_data = dataset.SeqDataset(
                data_dir,
                split_label=split,
                batch_size=batch_size,
                mode="eval",
                model_type=self.model_type,
            )
            
            # Extract sequences from dataset
            sequences_list = []
            targets_list = []
            # Determine how many batches to take
            max_batches = None if self.max_samples is None else (self.max_samples // batch_size + 1)
            dataset_to_use = seq_data.dataset if max_batches is None else seq_data.dataset.take(max_batches)
            
            for batch_data in dataset_to_use:
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    batch_sequences, batch_targets = batch_data
                    sequences_list.append(batch_sequences.numpy())
                    targets_list.append(batch_targets.numpy())
                else:
                    sequences_list.append(batch_data.numpy())
            
            if sequences_list:
                # Use CPU for large concatenation to avoid GPU OOM
                with tf.device('/CPU:0'):
                    data_tensor = tf.concat(sequences_list, axis=0)
                print(f"Loaded {data_tensor.shape[0]} sequences from TFRecord")
                if self.max_samples is not None:
                    return data_tensor[:self.max_samples]
                else:
                    return data_tensor
            
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        if not sequences:
            raise ValueError("No sequences loaded")
        
        print(f"Loaded {len(sequences)} sequences")
        
        # Limit number of sequences if max_samples is specified
        if self.max_samples is not None and len(sequences) > self.max_samples:
            print(f"Randomly sampling {self.max_samples} sequences from {len(sequences)}")
            indices = np.random.choice(len(sequences), self.max_samples, replace=False)
            sequences = [sequences[i] for i in indices]
        elif self.max_samples is None:
            print(f"Using all {len(sequences)} sequences for CKA analysis")
        
        # Convert sequences to one-hot encoding in batches to avoid memory issues
        print("Converting sequences to one-hot encoding...")
        data_list = []
        encoding_batch_size = 100  # Process sequences in smaller batches
        
        for batch_start in tqdm(range(0, len(sequences), encoding_batch_size), desc="Encoding sequence batches"):
            batch_end = min(batch_start + encoding_batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]
            
            batch_data = []
            for i, seq in enumerate(batch_sequences):
                try:
                    seq_1hot = dna.dna_1hot(seq)
                    batch_data.append(seq_1hot)
                except Exception as e:
                    print(f"Error encoding sequence {batch_start + i}: {e}")
                    continue
            
            if batch_data:
                data_list.extend(batch_data)
            
            # Periodic memory cleanup
            if batch_start % (encoding_batch_size * 10) == 0:
                import gc
                gc.collect()
        
        if not data_list:
            raise ValueError("No sequences could be encoded")
        
        print(f"Successfully encoded {len(data_list)} sequences")
        
        # Convert to tensor on CPU first to avoid GPU OOM during creation
        # Process in chunks if the array is very large
        max_chunk_size = 1000
        if len(data_list) > max_chunk_size:
            print(f"Processing large dataset in chunks of {max_chunk_size}...")
            tensor_chunks = []
            for chunk_start in tqdm(range(0, len(data_list), max_chunk_size), desc="Creating tensor chunks"):
                chunk_end = min(chunk_start + max_chunk_size, len(data_list))
                chunk_data = data_list[chunk_start:chunk_end]
                
                with tf.device('/CPU:0'):
                    chunk_tensor = tf.constant(np.array(chunk_data), dtype=tf.float32)
                tensor_chunks.append(chunk_tensor)
            
            # Concatenate all chunks
            with tf.device('/CPU:0'):
                data_tensor = tf.concat(tensor_chunks, axis=0)
        else:
            with tf.device('/CPU:0'):
                data_tensor = tf.constant(np.array(data_list), dtype=tf.float32)
        
        print(f"Final data tensor shape: {data_tensor.shape}")
        return data_tensor
    
    def select_layers(self, model: tf.keras.Model, layer_selection: str = 'auto',
                     custom_layers: List[str] = None) -> List[str]:
        """Select layers for CKA analysis.
        
        Args:
            model: TensorFlow model
            layer_selection: Layer selection strategy ('auto', 'all', 'conv', 'custom')
            custom_layers: Custom list of layer names (if layer_selection='custom')
            
        Returns:
            List of selected layer names
        """
        all_layers = [layer.name for layer in model.layers]
        
        if layer_selection == 'custom' and custom_layers:
            # Validate custom layers exist
            valid_layers = []
            for layer_name in custom_layers:
                if layer_name in all_layers:
                    valid_layers.append(layer_name)
                else:
                    print(f"Warning: Layer '{layer_name}' not found in model")
            return valid_layers
        
        elif layer_selection == 'all':
            return all_layers
        
        elif layer_selection == 'conv':
            # Select convolutional and dense layers
            selected = []
            for layer in model.layers:
                layer_type = type(layer).__name__.lower()
                if any(keyword in layer_type for keyword in ['conv', 'dense', 'attention']):
                    selected.append(layer.name)
            return selected
        
        elif layer_selection == 'auto':
            # Automatically select meaningful layers
            selected = []
            skip_types = ['input', 'dropout', 'reshape', 'flatten', 'lambda']
            
            for layer in model.layers:
                layer_type = type(layer).__name__.lower()
                
                # Skip layers that don't have meaningful representations
                if any(skip_type in layer_type for skip_type in skip_types):
                    continue
                
                # Include layers with activations or major computation
                if (hasattr(layer, 'activation') or 
                    any(keyword in layer_type for keyword in ['conv', 'dense', 'attention', 'norm'])):
                    selected.append(layer.name)
            
            return selected
        
        else:
            raise ValueError(f"Unknown layer selection strategy: {layer_selection}")
    
    def compute_cka_analysis(self, model1: tf.keras.Model, model2: tf.keras.Model = None,
                           data: tf.Tensor = None, layer_names1: List[str] = None,
                           layer_names2: List[str] = None, extraction_batch_size: int = 32,
                           max_samples_cka: int = 1500, use_streaming: bool = False, 
                           use_optimized_streaming: bool = True) -> Dict:
        """Compute comprehensive CKA analysis.
        
        Args:
            model1: First model
            model2: Second model (None for intra-model analysis only)
            data: Input data tensor
            layer_names1: Layer names for first model
            layer_names2: Layer names for second model
            extraction_batch_size: Batch size for activation extraction or streaming computation
            use_streaming: Whether to use streaming online CKA computation (saves memory)
            use_optimized_streaming: Whether to use optimized streaming with multi-output models
            
        Returns:
            Dictionary containing analysis results
        """
        if data is None:
            raise ValueError("Data tensor is required for CKA analysis")
        
        if use_streaming:
            # Use streaming online CKA computation - no activation storage
            if use_optimized_streaming:
                print("Using optimized streaming CKA computation for maximum efficiency...")
            else:
                print("Using basic streaming online CKA computation for memory efficiency...")
                
            if model2 is None:
                # Intra-model analysis only
                print("Computing streaming intra-model CKA analysis...")
                intra_cka1, layer_names_final, _ = self.cka_analyzer.compute_cka_matrix_streaming(
                    model1, None, data, layer_names1, layer_names1, extraction_batch_size, use_optimized_streaming
                )
                
                results = {
                    'intra_model1_cka': intra_cka1,
                    'layer_names1': layer_names_final,
                    'layer_names2': layer_names_final,
                    'analysis_type': 'optimized_streaming_intra' if use_optimized_streaming else 'streaming_intra_model'
                }
            else:
                # Full inter-model and intra-model analysis using streaming
                print("Computing comprehensive streaming CKA analysis...")
                
                # Inter-model CKA
                print("Computing inter-model CKA...")
                inter_cka, layer_names1_final, layer_names2_final = self.cka_analyzer.compute_cka_matrix_streaming(
                    model1, model2, data, layer_names1, layer_names2, extraction_batch_size, use_optimized_streaming
                )
                
                # Intra-model CKA for model 1
                print("Computing intra-model CKA for model 1...")
                intra_cka1, _, _ = self.cka_analyzer.compute_cka_matrix_streaming(
                    model1, None, data, layer_names1_final, layer_names1_final, extraction_batch_size, use_optimized_streaming
                )
                
                # Intra-model CKA for model 2
                print("Computing intra-model CKA for model 2...")
                intra_cka2, _, _ = self.cka_analyzer.compute_cka_matrix_streaming(
                    model2, None, data, layer_names2_final, layer_names2_final, extraction_batch_size, use_optimized_streaming
                )
                
                results = {
                    'inter_model_cka': inter_cka,
                    'intra_model1_cka': intra_cka1,
                    'intra_model2_cka': intra_cka2,
                    'layer_names1': layer_names1_final,
                    'layer_names2': layer_names2_final,
                    'analysis_type': 'optimized_streaming_inter' if use_optimized_streaming else 'streaming_inter_model'
                }
        else:
            # Use traditional method with activation storage
            print("Using traditional CKA computation with activation storage...")
            if model2 is None:
                # Intra-model analysis only
                print("Computing intra-model CKA analysis...")
                activations1 = self.cka_analyzer.extract_layer_activations(model1, data, layer_names1, extraction_batch_size, self.use_memmap, self.use_fast_extraction)
                intra_cka1, layer_names_final, _ = self.cka_analyzer.compute_cka_matrix(activations1)
                
                results = {
                    'intra_model1_cka': intra_cka1,
                    'layer_names1': layer_names_final,
                    'layer_names2': layer_names_final,
                    'activations1': activations1,
                    'analysis_type': 'intra_model'
                }
            else:
                # Full inter-model and intra-model analysis
                print("Computing comprehensive CKA analysis...")
                results = self.cka_analyzer.analyze_models(
                    model1, model2, data, layer_names1, layer_names2, extraction_batch_size, self.use_memmap, self.use_fast_extraction
                )
                results['analysis_type'] = 'inter_model'
        
        return results
    
    def generate_report(self, results: Dict, output_dir: str):
        """Generate comprehensive analysis report.
        
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
  # Compare two models using BED file data
  python models_cka_compute.py \\
    --model1_params model1.yaml --model1_weights model1.h5 \\
    --model2_params model2.yaml --model2_weights model2.h5 \\
    --data_source bed --bed_file sequences.bed --genome_fasta genome.fa \\
    --output_dir cka_results

  # Analyze single model with direct sequences
  python models_cka_compute.py \\
    --model1_params model.yaml --model1_weights model.h5 \\
    --data_source sequences --sequences ATCGATCG GCTAGCTA \\
    --output_dir cka_results --intra_only

  # Use TFRecord dataset
  python models_cka_compute.py \\
    --model1_params model1.yaml --model1_weights model1.h5 \\
    --model2_params model2.yaml --model2_weights model2.h5 \\
    --data_source tfrecord --data_dir /path/to/tfrecords \\
    --output_dir cka_results

  # Use streaming CKA for memory efficiency (recommended for large models)
  python models_cka_compute.py \\
    --model1_params model1.yaml --model1_weights model1.h5 \\
    --model2_params model2.yaml --model2_weights model2.h5 \\
    --data_source bed --bed_file sequences.bed --genome_fasta genome.fa \\
    --output_dir cka_results --use_streaming --streaming_batch_size 8
        """
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model1_params", required=True, 
                           help="YAML parameter file for first model")
    model_group.add_argument("--model1_weights", required=True,
                           help="Weights file for first model")
    model_group.add_argument("--model2_params",
                           help="YAML parameter file for second model")
    model_group.add_argument("--model2_weights", 
                           help="Weights file for second model")
    
    # Data source arguments
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--data_source", required=True,
                          choices=['bed', 'fasta', 'sequences', 'tfrecord'],
                          help="Type of data source")
    data_group.add_argument("--bed_file", help="BED file with genomic coordinates")
    data_group.add_argument("--genome_fasta", help="Genome FASTA file")
    data_group.add_argument("--fasta_file", help="FASTA file with sequences")
    data_group.add_argument("--sequences", nargs='+', help="Direct sequence inputs")
    data_group.add_argument("--data_dir", help="TFRecord data directory")
    data_group.add_argument("--seq_length", type=int, default=32768,
                          help="Sequence length for BED extraction")
    
    # Layer selection arguments
    layer_group = parser.add_argument_group("Layer Selection")
    layer_group.add_argument("--layer_selection", choices=['auto', 'all', 'conv', 'custom'],
                           default='auto', help="Layer selection strategy")
    layer_group.add_argument("--model1_layers", nargs='+', 
                           help="Custom layer names for model 1")
    layer_group.add_argument("--model2_layers", nargs='+',
                           help="Custom layer names for model 2")
    
    # Analysis arguments
    analysis_group = parser.add_argument_group("Analysis Configuration")
    analysis_group.add_argument("--kernel", choices=['linear', 'rbf'], default='linear',
                              help="Kernel type for CKA computation")
    analysis_group.add_argument("--batch_size", type=int, default=16,
                              help="Batch size for processing")
    analysis_group.add_argument("--extraction_batch_size", type=int, default=None,
                              help="Batch size for activation extraction (default: auto-optimize based on data size)")
    analysis_group.add_argument("--use_memmap", action='store_true',
                              help="Use memory mapping for activations to reduce RAM usage (recommended for large datasets)")
    analysis_group.add_argument("--use_fast_extraction", action='store_true', default=True,
                              help="Use static graph + GPU optimized extraction (default: True)")
    analysis_group.add_argument("--max_samples", type=int, default=None,
                              help="Maximum number of samples to use for CKA analysis (default: use all samples)")
    analysis_group.add_argument("--intra_only", action='store_true',
                              help="Only perform intra-model analysis")
    analysis_group.add_argument("--use_streaming", action='store_true',
                              help="Use streaming online CKA computation to save memory (does not store activations)")
    analysis_group.add_argument("--streaming_batch_size", type=int, default=16,
                              help="Batch size for streaming CKA computation (smaller = less memory)")
    analysis_group.add_argument("--use_optimized_streaming", action='store_true', default=True,
                              help="Use optimized streaming with multi-output models (much faster, default: True)")
    analysis_group.add_argument("--random_seed", type=int, default=42,
                              help="Random seed for reproducibility")
    
    # Output arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--output_dir", default="cka_output",
                            help="Output directory for results")
    output_group.add_argument("--save_activations", action='store_true',
                            help="Save extracted activations")
    output_group.add_argument("--no_plots", action='store_true',
                            help="Skip generating plots")
    output_group.add_argument("--no_report", action='store_true',
                            help="Skip generating text report")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.intra_only and (not args.model2_params or not args.model2_weights):
        parser.error("model2_params and model2_weights required unless --intra_only is specified")
    
    # Data source validation
    if args.data_source == 'bed' and (not args.bed_file or not args.genome_fasta):
        parser.error("--bed_file and --genome_fasta required for BED data source")
    elif args.data_source == 'fasta' and not args.fasta_file:
        parser.error("--fasta_file required for FASTA data source")
    elif args.data_source == 'sequences' and not args.sequences:
        parser.error("--sequences required for direct sequence input")
    elif args.data_source == 'tfrecord' and not args.data_dir:
        parser.error("--data_dir required for TFRecord data source")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize CKA computer
    cka_computer = ModelCKAComputer(
        kernel=args.kernel,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        random_seed=args.random_seed,
        extraction_batch_size=args.extraction_batch_size,
        use_memmap=args.use_memmap,
        use_fast_extraction=args.use_fast_extraction
    )
    
    try:
        # Load models
        print("Loading models...")
        model1 = cka_computer.load_model(args.model1_params, args.model1_weights, "Model 1")
        
        model2 = None
        if not args.intra_only:
            model2 = cka_computer.load_model(args.model2_params, args.model2_weights, "Model 2")
        
        # Load data
        print("Loading data...")
        data_kwargs = {
            'bed_file': args.bed_file,
            'genome_fasta': args.genome_fasta,
            'fasta_file': args.fasta_file,
            'sequences': args.sequences,
            'data_dir': args.data_dir,
            'seq_length': args.seq_length,
            'batch_size': args.batch_size
        }
        data = cka_computer.load_data(args.data_source, **data_kwargs)
        
        # Select layers
        print("Selecting layers for analysis...")
        layer_names1 = cka_computer.select_layers(
            model1, args.layer_selection, args.model1_layers
        )
        print(f"Selected {len(layer_names1)} layers from Model 1")
        
        layer_names2 = None
        if model2 is not None:
            layer_names2 = cka_computer.select_layers(
                model2, args.layer_selection, args.model2_layers
            )
            print(f"Selected {len(layer_names2)} layers from Model 2")
        
        # Compute CKA analysis
        print("Computing CKA analysis...")
        
        # Determine batch size for extraction or streaming
        if args.use_streaming:
            batch_size = args.streaming_batch_size
            print(f"Using streaming CKA with batch size: {batch_size}")
        else:
            batch_size = args.extraction_batch_size
            if batch_size:
                print(f"Using specified extraction batch size: {batch_size}")
            else:
                print("Using auto-optimized extraction batch size")
        
        results = cka_computer.compute_cka_analysis(
            model1, model2, data, layer_names1, layer_names2, 
            batch_size, use_streaming=args.use_streaming,
            use_optimized_streaming=args.use_optimized_streaming
        )
        
        # Save results
        print("Saving results...")
        results_path = os.path.join(args.output_dir, "cka_results.h5")
        cka_computer.cka_analyzer.save_results(results, results_path)
        
        # Generate report
        if not args.no_report:
            cka_computer.generate_report(results, args.output_dir)
        
        # Generate plots
        if not args.no_plots:
            cka_computer.generate_plots(results, args.output_dir)
        
        # Save activations if requested (only available for non-streaming method)
        if args.save_activations:
            if args.use_streaming:
                print("Warning: Cannot save activations when using streaming CKA method")
            else:
                print("Saving activations...")
                activations_path = os.path.join(args.output_dir, "activations.h5")
                with h5py.File(activations_path, 'w') as f:
                    if 'activations1' in results:
                        act1_group = f.create_group('model1_activations')
                        for layer_name, activation in results['activations1'].items():
                            act1_group.create_dataset(layer_name, data=activation.numpy())
                    
                    if 'activations2' in results:
                        act2_group = f.create_group('model2_activations')
                        for layer_name, activation in results['activations2'].items():
                            act2_group.create_dataset(layer_name, data=activation.numpy())
        
        print(f"\\n*** CKA Analysis Complete! ***")
        print(f"*** Results saved to: {args.output_dir} ***")
        print(f"*** Main results file: {results_path} ***")
        
        # Print quick summary
        if 'inter_model_cka' in results:
            inter_mean = np.mean(results['inter_model_cka'])
            print(f"*** Inter-model CKA mean similarity: {inter_mean:.4f} ***")
        
        if 'intra_model1_cka' in results:
            intra_mean = np.mean(results['intra_model1_cka'][np.triu_indices_from(results['intra_model1_cka'], k=1)])
            print(f"*** Intra-model CKA mean similarity: {intra_mean:.4f} ***")
        
    except Exception as e:
        print(f"Error during CKA analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
