#!/usr/bin/env python

import argparse
import os
import sys
import yaml
import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd

import paddy.utils as utils
import random

# Set GPU device(s) before importing TensorFlow
gpu_ids = utils.set_gpu_device()

import tensorflow as tf
from paddy import dataset
from paddy import seqnn
from paddy import trainer
from paddy import layers
from paddy import transfer
from paddy import bed
from paddy import dna
from paddy import embed

"""
paddy_embed.py
Enhanced embedding extraction tool for sequence models.
Supports multiple layer selection methods and embedding extraction strategies.
"""

def list_model_layers(model, verbose=True):
    """List all layers in the model with detailed information."""
    layers_info = []
    
    if verbose:
        print("\n=== Model Layer Information ===")
        print(f"{'Index':<6} {'Layer Name':<40} {'Layer Type':<25} {'Output Shape':<20}")
        print("-" * 95)
    
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        try:
            output_shape = str(layer.output_shape)
        except:
            output_shape = "Unknown"
        
        layers_info.append({
            'index': i,
            'name': layer.name,
            'type': layer_type,
            'output_shape': output_shape,
            'layer': layer
        })
        
        if verbose:
            print(f"{i:<6} {layer.name:<40} {layer_type:<25} {output_shape:<20}")
    
    return layers_info

def find_layers_by_type(model, layer_types):
    """Find layers by their type(s)."""
    if isinstance(layer_types, str):
        layer_types = [layer_types]
    
    found_layers = []
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        for target_type in layer_types:
            if target_type.lower() in layer_type.lower():
                found_layers.append({
                    'index': i,
                    'name': layer.name,
                    'type': layer_type,
                    'output_shape': layer.output_shape,
                    'layer': layer
                })
                break
    
    return found_layers

def build_custom_embed_model(base_model, layer_index=None, layer_name=None):
    """Build embedding model from specific layer."""
    if layer_index is not None:
        if layer_index == -1:
            # Use the final model output
            target_layer = base_model.layers[-1]
        else:
            target_layer = base_model.layers[layer_index]
    elif layer_name is not None:
        target_layer = None
        for layer in base_model.layers:
            if layer.name == layer_name:
                target_layer = layer
                break
        if target_layer is None:
            raise ValueError(f"Layer with name '{layer_name}' not found")
    else:
        raise ValueError("Either layer_index or layer_name must be specified")
    
    # Create embedding model
    embed_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=target_layer.output,
        name=f"embed_model_{target_layer.name}"
    )
    
    return embed_model

def extract_embeddings_from_sequences(model, sequences, batch_size=32, verbose=True):
    """Extract embeddings from sequences."""
    if verbose:
        print(f"\nExtracting embeddings from {len(sequences)} sequences...")
        print(f"Batch size: {batch_size}")
        print(f"Expected output shape: {model.output_shape}")
    
    embeddings = []
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    iterator = tqdm(range(num_batches), desc="Processing batches") if verbose else range(num_batches)
    
    for i in iterator:
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        # Convert sequences to one-hot encoding
        batch_1hot = np.array([dna.dna_1hot(seq) for seq in batch_sequences])
        
        # Extract embeddings
        batch_embeddings = model(batch_1hot).numpy()
        embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings, axis=0)
    
    if verbose:
        print(f"Final embedding shape: {all_embeddings.shape}")
    
    return all_embeddings

def save_embeddings(embeddings, output_path, sequences_info=None, layer_info=None):
    """Save embeddings to HDF5 file with metadata."""
    print(f"\nSaving embeddings to: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Save embeddings
        f.create_dataset('embeddings', data=embeddings, compression='gzip')
        
        # Save metadata
        if sequences_info is not None:
            if 'coordinates' in sequences_info:
                coords = sequences_info['coordinates']
                f.create_dataset('coordinates/chrom', data=[c[0].encode() for c in coords])
                f.create_dataset('coordinates/start', data=[c[1] for c in coords])
                f.create_dataset('coordinates/end', data=[c[2] for c in coords])
            
            if 'sequences' in sequences_info:
                f.create_dataset('sequences', data=[s.encode() for s in sequences_info['sequences']])
        
        # Save layer info
        if layer_info is not None:
            f.attrs['layer_name'] = layer_info.get('name', 'unknown')
            f.attrs['layer_type'] = layer_info.get('type', 'unknown')
            f.attrs['layer_index'] = layer_info.get('index', -1)
            f.attrs['output_shape'] = str(layer_info.get('output_shape', 'unknown'))
        
        # Save shape info
        f.attrs['embedding_shape'] = embeddings.shape
        f.attrs['num_sequences'] = embeddings.shape[0]
    
    print(f"Saved embeddings with shape {embeddings.shape}")

def interactive_layer_selection(model):
    """Interactive layer selection for users."""
    layers_info = list_model_layers(model, verbose=True)
    
    print("\n=== Layer Selection Options ===")
    print("1. Select by layer index")
    print("2. Select by layer name")
    print("3. Find layers by type")
    print("4. Use trunk output (-1)")
    print("5. Browse specific layer types")
    
    while True:
        try:
            choice = input("\nChoose an option (1-5): ").strip()
            
            if choice == '1':
                index = int(input("Enter layer index: "))
                if 0 <= index < len(layers_info) or index == -1:
                    return index, None
                else:
                    print(f"Invalid index. Must be between 0 and {len(layers_info)-1}, or -1 for trunk output.")
            
            elif choice == '2':
                name = input("Enter layer name: ").strip()
                for layer_info in layers_info:
                    if layer_info['name'] == name:
                        return layer_info['index'], name
                print(f"Layer name '{name}' not found.")
            
            elif choice == '3':
                layer_type = input("Enter layer type (e.g., 'conv', 'batch', 'dense'): ").strip()
                found = find_layers_by_type(model, layer_type)
                if found:
                    print(f"\nFound {len(found)} layers of type '{layer_type}':")
                    for i, layer_info in enumerate(found):
                        print(f"  {i}: Index {layer_info['index']} - {layer_info['name']} ({layer_info['type']})")
                    
                    selection = int(input(f"Select which one (0-{len(found)-1}): "))
                    if 0 <= selection < len(found):
                        selected = found[selection]
                        return selected['index'], selected['name']
                else:
                    print(f"No layers found with type '{layer_type}'")
            
            elif choice == '4':
                return -1, None
            
            elif choice == '5':
                print("\nCommon layer types:")
                print("- Conv layers:", len([l for l in layers_info if 'conv' in l['type'].lower()]))
                print("- BatchNorm layers:", len([l for l in layers_info if 'batch' in l['type'].lower()]))
                print("- Dense layers:", len([l for l in layers_info if 'dense' in l['type'].lower()]))
                print("- Attention layers:", len([l for l in layers_info if 'attention' in l['type'].lower()]))
                print("- Activation layers:", len([l for l in layers_info if 'activation' in l['type'].lower()]))
                
                layer_type = input("\nEnter layer type to explore: ").strip()
                found = find_layers_by_type(model, layer_type)
                if found:
                    print(f"\nFound {len(found)} layers:")
                    for i, layer_info in enumerate(found):
                        print(f"  {i}: Index {layer_info['index']} - {layer_info['name']} - {layer_info['output_shape']}")
                    
                    selection = int(input(f"Select which one (0-{len(found)-1}): "))
                    if 0 <= selection < len(found):
                        selected = found[selection]
                        return selected['index'], selected['name']
            
            else:
                print("Invalid choice. Please enter 1-5.")
                
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            sys.exit(0)

def analyze_single_embedding(embed_file, output_dir, args):
    """Analyze a single embedding file."""
    print(f"\n=== Analyzing Embedding File: {embed_file} ===")
    
    # Load embeddings
    embeddings, metadata, coordinates, sequences = embed.load_embeddings(embed_file)
    
    # Basic analysis
    stats_dict = embed.analyze_embeddings(embeddings, metadata, verbose=True)
    
    # Quality assessment
    if args.quality_check or True:  # Always do quality check in analysis mode
        quality_scores, recommendations = embed.evaluate_embedding_quality(embeddings, verbose=True)
    
    if not args.no_plots:
        print(f"\nGenerating analysis plots...")
        
        # Distribution plots
        embed.plot_embedding_distribution(embeddings, output_dir, title_suffix=" Analysis")
        
        # Dimensionality reduction
        if embeddings.shape[0] > 1:
            reduced, reducer = embed.reduce_dimensions(embeddings, method=args.dim_reduction)
            if reduced is not None and reducer is not None:
                embed.plot_2d_embeddings(reduced, coordinates, output_dir, 
                                        args.dim_reduction, title_suffix=" Analysis")
            
            # Similarity analysis
            similarity = embed.compute_similarity_matrix(embeddings, args.similarity_metric)
            embed.plot_similarity_matrix(similarity, output_dir, 
                                       args.similarity_metric, title_suffix=" Analysis")
            
            # Clustering analysis
            labels, silhouette_score = embed.cluster_embeddings(embeddings)
            print(f"Clustering silhouette score: {silhouette_score:.4f}")
        else:
            print("Only one sequence found, skipping multi-sequence analysis.")
    
    # Save analysis report
    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("Embedding Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"File: {embed_file}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n\n")
        
        f.write("Basic Statistics:\n")
        for key, value in stats_dict.items():
            f.write(f"  {key}: {value}\n")
        
        if 'quality_scores' in locals():
            f.write(f"\nQuality Assessment:\n")
            for key, value in quality_scores.items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write(f"\nRecommendations:\n")
            for rec in recommendations:
                f.write(f"  {rec}\n")
    
    print(f"\nAnalysis completed! Results saved to: {output_dir}")
    print(f"Detailed report: {report_path}")

def compare_multiple_embeddings(embed_files, output_dir, args):
    """Compare multiple embedding files."""
    print(f"\n=== Comparing {len(embed_files)} Embedding Files ===")
    
    # Validate files exist
    for file_path in embed_files:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    # Perform comparison
    comparison_results = embed.compare_embeddings(embed_files, output_dir, verbose=True)
    
    # Generate comparison plots if requested
    if not args.no_plots:
        print(f"\n Generating comparison plots...")
        # Additional comparison plots can be added here
    
    print(f"\nComparison completed! Results saved to: {output_dir}")
    
    # Print quality ranking summary
    print(f"\nQuality Ranking Summary:")
    quality_ranking = sorted(
        comparison_results.items(),
        key=lambda x: x[1]['quality']['overall'],
        reverse=True
    )
    
    for rank, (key, results) in enumerate(quality_ranking, 1):
        quality = results['quality']['overall']
        file_name = os.path.basename(results['file_path'])
        layer_name = results.get('metadata', {}).get('layer_name', 'unknown')
        print(f"  {rank}. {file_name}: {quality:.3f} (layer: {layer_name})")

def extraction_mode(args, parser):
    """Handle embedding extraction mode."""
    # Parse shifts
    args.shifts = [int(shift) for shift in args.shifts.split(",")]
    
    # Load model parameters and create model
    print("Loading model...")
    with open(args.params_file) as params_open:
        params = yaml.safe_load(params_open)
    params_model = params["model"]
    params_model["verbose"] = False
    
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(args.model_file, trunk=True)
    
    # Apply ensemble if requested
    if args.ensemble_rc:
        seqnn_model.build_ensemble(True, args.shifts)
    
    # List layers and exit if requested
    if args.list_layers:
        list_model_layers(seqnn_model.model, verbose=True)
        return
    
    # Find layers by type if requested
    if args.layer_type:
        found_layers = find_layers_by_type(seqnn_model.model, args.layer_type)
        print(f"\nFound {len(found_layers)} layers of type '{args.layer_type}':")
        for layer_info in found_layers:
            print(f"  Index {layer_info['index']}: {layer_info['name']} - {layer_info['type']} - {layer_info['output_shape']}")
            return
    
    # Validate input data
    if not any([args.bed_file, args.fasta_file, args.sequences]):
        parser.error("Must specify one of: --bed_file, --fasta_file, or --sequences")
    
    # Determine which layer to use for embedding
    layer_index = None
    layer_name = None
    layer_info = None
    
    if args.interactive:
        layer_index, layer_name = interactive_layer_selection(seqnn_model.model)
    elif args.layer_index is not None:
        layer_index = args.layer_index
    elif args.layer_name is not None:
        layer_name = args.layer_name
    elif args.embed_layer is not None:
        # Legacy support
        print("Using legacy embed_layer parameter...")
        seqnn_model.build_embed(args.embed_layer, args.batch_norm)
        embed_model = seqnn_model.model
    else:
        print("No layer specified. Use --interactive for interactive selection, or specify --layer_index/--layer_name")
        print("Use --list_layers to see all available layers")
        return
    
    # Build embedding model if not using legacy method
    if args.embed_layer is None:
        embed_model = build_custom_embed_model(seqnn_model.model, layer_index, layer_name)
        
        # Get layer info for metadata
        if layer_index is not None:
            if layer_index == -1:
                target_layer = seqnn_model.model.layers[-1]
                layer_info = {
                    'index': len(seqnn_model.model.layers) - 1,
                    'name': target_layer.name,
                    'type': type(target_layer).__name__,
                    'output_shape': target_layer.output_shape
                }
            else:
                target_layer = seqnn_model.model.layers[layer_index]
                layer_info = {
                    'index': layer_index,
                    'name': target_layer.name,
                    'type': type(target_layer).__name__,
                    'output_shape': target_layer.output_shape
                }
        elif layer_name is not None:
            for i, layer in enumerate(seqnn_model.model.layers):
                if layer.name == layer_name:
                    layer_info = {
                        'index': i,
                        'name': layer.name,
                        'type': type(layer).__name__,
                        'output_shape': layer.output_shape
                    }
                    break
    
    print(f"\nUsing embedding model with output shape: {embed_model.output_shape}")
    if layer_info:
        print(f"Layer: {layer_info['name']} (index {layer_info['index']}, type {layer_info['type']})")
    
    # Load sequences
    sequences = []
    sequences_info = {}
    
    if args.bed_file:
        print(f"Loading sequences from BED file: {args.bed_file}")
        if not args.genome_fasta:
            raise ValueError("--genome_fasta required when using BED file")
        
        seq_length = args.seq_length or params_model.get('seq_length', 32768)
        model_sequences, model_coords = bed.make_bed_seqs(
            args.bed_file, args.genome_fasta, seq_length, stranded=False
        )
        sequences = model_sequences
        sequences_info['coordinates'] = model_coords
        
    elif args.fasta_file:
        print(f"Loading sequences from FASTA file: {args.fasta_file}")
        # Simple FASTA reader
        sequences = []
        with open(args.fasta_file, 'r') as f:
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
        sequences_info['sequences'] = sequences
        
    elif args.sequences:
        print(f"Using {len(args.sequences)} direct sequence inputs")
        sequences = [seq.upper() for seq in args.sequences]
        sequences_info['sequences'] = sequences
        
    else:
        raise ValueError("Must specify one of: --bed_file, --fasta_file, or --sequences")
    
    print(f"Loaded {len(sequences)} sequences")
    if sequences:
        print(f"First sequence length: {len(sequences[0])}")
    
    # Extract embeddings
    embeddings = extract_embeddings_from_sequences(
        embed_model, sequences, batch_size=args.batch_size, verbose=True
    )
    
    # Save embeddings
    output_path = os.path.join(args.out_dir, args.output_name)
    save_embeddings(embeddings, output_path, sequences_info, layer_info)
    
    print(f"\n*** Embedding extraction completed! ***")
    print(f"*** Output saved to: {output_path} ***")
    print(f"*** Embedding shape: {embeddings.shape} ***")
    if layer_info:
        print(f"*** Source layer: {layer_info['name']} (index {layer_info['index']}) ***")
    
    # Auto-analyze if requested
    if args.auto_analyze:
        print(f"\nAuto-analyzing extracted embeddings...")
        analyze_single_embedding(output_path, args.out_dir, args)

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced embedding extraction and analysis tool for sequence models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract embeddings interactively
  python paddy_embed.py params.yaml model.h5 --bed_file sequences.bed --genome_fasta genome.fa --interactive

  # Analyze existing embeddings
  python paddy_embed.py --analyze embeddings.h5 --output_dir analysis_results

  # Compare multiple embeddings
  python paddy_embed.py --compare embed1.h5 embed2.h5 embed3.h5 --output_dir comparison_results

  # Extract and analyze in one go
  python paddy_embed.py params.yaml model.h5 --layer_index -1 --bed_file sequences.bed --genome_fasta genome.fa --auto_analyze
        """
    )
    
    # Create mutually exclusive groups
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("params_file", nargs='?', help="Parameters file for extraction mode")
    mode_group.add_argument("--analyze", metavar="EMBED_FILE", help="Analyze existing embedding file")
    mode_group.add_argument("--compare", nargs='+', metavar="EMBED_FILES", help="Compare multiple embedding files")
    
    # Model file (required for extraction mode)
    parser.add_argument("model_file", nargs='?', help="Model file (required for extraction mode)")
    
    # Input data arguments (for extraction mode)
    input_group = parser.add_argument_group("Input Data (extraction mode)")
    input_data_group = input_group.add_mutually_exclusive_group()
    input_data_group.add_argument("-b", "--bed_file", help="BED file with sequences")
    input_data_group.add_argument("-f", "--fasta_file", help="FASTA file with sequences")
    input_data_group.add_argument("--sequences", nargs="+", help="Direct sequence inputs")
    
    # Layer selection arguments (for extraction mode)
    layer_group = parser.add_argument_group("Layer Selection (extraction mode)")
    layer_selection_group = layer_group.add_mutually_exclusive_group()
    layer_selection_group.add_argument("-l", "--layer_index", type=int, help="Layer index for embedding extraction")
    layer_selection_group.add_argument("--layer_name", help="Layer name for embedding extraction")
    layer_selection_group.add_argument("--interactive", action="store_true", help="Interactive layer selection")
    layer_selection_group.add_argument("--embed_layer", type=int, help="Legacy: conv layer index for embedding")
    
    layer_group.add_argument("--list_layers", action="store_true", help="List all model layers and exit")
    layer_group.add_argument("--layer_type", help="Find layers by type (e.g., 'conv', 'batch')")
    layer_group.add_argument("--batch_norm", action="store_true", default=True, help="Legacy: use batch norm layer")
    
    # Processing arguments
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    process_group.add_argument("--ensemble_rc", action="store_true", help="Use reverse complement ensemble")
    process_group.add_argument("--shifts", default="0", help="Ensemble prediction shifts")
    process_group.add_argument("--seq_length", type=int, help="Sequence length (auto-detected if not provided)")
    process_group.add_argument("--genome_fasta", help="Genome FASTA file for BED sequences")
    
    # Output arguments
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("-o", "--out_dir", "--output_dir", dest="out_dir", default="embed_out", help="Output directory")
    output_group.add_argument("--output_name", default="embeddings.h5", help="Output file name")
    
    # Analysis arguments
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument("--auto_analyze", action="store_true", help="Automatically analyze after extraction")
    analysis_group.add_argument("--no_plots", action="store_true", help="Skip plotting (analysis mode)")
    analysis_group.add_argument("--dim_reduction", choices=['pca', 'tsne', 'umap'], default='pca', help="Dimensionality reduction method")
    analysis_group.add_argument("--similarity_metric", choices=['cosine', 'euclidean'], default='cosine', help="Similarity metric")
    analysis_group.add_argument("--quality_check", action="store_true", help="Perform embedding quality assessment")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Handle different modes
    if args.analyze:
        # Analysis mode
        print(f"Analysis Mode: {args.analyze}")
        return analyze_single_embedding(args.analyze, args.out_dir, args)
    
    elif args.compare:
        # Comparison mode
        print(f"Comparison Mode: {len(args.compare)} files")
        return compare_multiple_embeddings(args.compare, args.out_dir, args)
    
    else:
        # Extraction mode
        if not args.params_file or not args.model_file:
            parser.error("params_file and model_file are required for extraction mode")
        
        print(f"Extraction Mode: {args.params_file}, {args.model_file}")
        return extraction_mode(args, parser)

if __name__ == "__main__":
    main()