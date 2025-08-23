#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import h5py
from scipy import ndimage
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess gradients for TF-MoDISco analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )  
    parser.add_argument(
        "--grad_dir",
        required=True,
        help="Directory containing tissues' gradients and input*gradients"
    )
    parser.add_argument(
        "--tissue_indices",
        required=True,
        help="Comma-separated tissue indices to process (e.g., 18,0,22,19,20, same indexing as in gradient file)"
    )
    parser.add_argument(
        "-o", "--out_dir",
        default="modisco_preprocessed",
        help="Output directory for preprocessed gradients"
    )
    parser.add_argument(
        "--center_size",
        type=int,
        default=32768,  # 32 kb
        help="Size of centered region to extract (in bp)"
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=None,
        help="Standard deviation for Gaussian filter. If not specified, no Gaussian filtering is applied."
    )
    parser.add_argument(
        "--gaussian_truncate",
        type=float,
        default=None,
        help="Truncate parameter for Gaussian filter. Only used if --gaussian_sigma is specified."
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Compute residual gradients (tissue - avg(others)). If not set, use raw gradients."
    )
    return parser.parse_args()

def load_gradient_data(grad_file, center_size=None):
    """Load gradient data from HDF5 file with optional centering."""
    print(f"Loading gradients from {grad_file}...")
    
    with h5py.File(grad_file, 'r') as f:
        # Get data info first
        grads_shape = f['grads'].shape
        seqs_shape = f['seqs'].shape
        
        print(f"  File gradients shape: {grads_shape}")
        print(f"  File sequences shape: {seqs_shape}")
        
        # Convert float16 to float32 to avoid computation issues
        grads_dtype = f['grads'].dtype
        if grads_dtype == np.float16:
            print("  Converting float16 to float32 for compatibility...")
            target_dtype = np.float32
        else:
            target_dtype = grads_dtype
        
        # Load with optional centering to save memory
        if center_size is not None and center_size < grads_shape[1]:
            seq_len = grads_shape[1]
            start = (seq_len - center_size) // 2
            end = start + center_size
            print(f"  Loading centered region: {start}:{end} (size: {center_size})")
            
            grads = f['grads'][:, start:end, :, :].astype(target_dtype)
            seqs = f['seqs'][:, start:end, :].astype(np.float32)
        else:
            grads = f['grads'][:].astype(target_dtype)
            seqs = f['seqs'][:].astype(np.float32)
        
        gene_ids = f['gene'][:] if 'gene' in f else None
      
    return grads, seqs, gene_ids

def extract_centered_region(data, center_size):
    """Extract centered region from gradient data."""
    seq_len = data.shape[1]
    
    if center_size >= seq_len:
        print(f"Warning: center_size ({center_size}) >= seq_len ({seq_len}), using full sequence")
        return data
    
    # Calculate start and end positions for centering
    start = (seq_len - center_size) // 2
    end = start + center_size
    
    print(f"Extracting centered region: positions {start} to {end} (size: {center_size})")
    
    return data[:, start:end, ...]

def compute_residual_gradients_efficient(tissue_grads, other_tissues_files, center_size):
    """Compute residual gradients efficiently by streaming other tissues."""
    print("Computing residual gradients (memory efficient)...")
    
    # Initialize accumulator for other tissues average
    other_sum = None
    other_count = 0
    
    # Stream through other tissue files to compute average
    for other_file in other_tissues_files:
        print(f"  Processing background tissue: {os.path.basename(other_file)}")
        other_grads, _, _ = load_gradient_data(other_file, center_size)
        
        if other_sum is None:
            other_sum = other_grads.astype(np.float64)  # Use float64 for accumulation
        else:
            other_sum += other_grads.astype(np.float64)
        other_count += 1
        
        # Free memory immediately
        del other_grads
    
    # Compute average
    other_avg = (other_sum / other_count).astype(tissue_grads.dtype)
    del other_sum
    
    # Subtract average from target tissue
    residual_grads = tissue_grads - other_avg
    
    print(f"  Residual gradients shape: {residual_grads.shape}")
    print(f"  Residual gradients range: {residual_grads.min():.4f} to {residual_grads.max():.4f}")
    
    return residual_grads

def compute_reweighting_factors(grads, gaussian_sigma, gaussian_truncate):
    """Compute per-position re-weighting factors.

    Expected method: std across 4 nucleotides at each position, averaged across genes,
    then optionally smoothed by a Gaussian filter.
    
    Args:
        grads: Gradient array (n_genes, seq_len, 4) or (n_genes, seq_len, 4, n_tissues)
        gaussian_sigma: Gaussian filter sigma
        gaussian_truncate: Gaussian filter truncate parameter
        tissue_idx: If provided and grads is 4D, use only this tissue for reweighting
    """
    print("Computing re-weighting factors...")

    print(f"  Input gradients shape: {grads.shape}")

    print("  Computing position-wise standard deviation across nucleotides...")

    chunk_size = min(1000, grads.shape[0])
    position_stds = []

    for i in range(0, grads.shape[0], chunk_size):
        end_idx = min(i + chunk_size, grads.shape[0])
        chunk = grads[i:end_idx]  # (chunk_size, seq_len, 4)
        chunk_std = np.std(chunk, axis=2)  # (chunk_size, seq_len)
        chunk_position_std = np.mean(chunk_std, axis=0)  # (seq_len,)
        position_stds.append(chunk_position_std)

    position_std = np.mean(position_stds, axis=0)  # (seq_len,)

    print(f"  Position std shape: {position_std.shape}")

    if gaussian_sigma is not None:
        print(f"  Applying Gaussian filter (sigma={gaussian_sigma}, truncate={gaussian_truncate})")
        smoothed_std = ndimage.gaussian_filter1d(
            position_std,
            sigma=gaussian_sigma,
            truncate=gaussian_truncate if gaussian_truncate is not None else 4.0,
        )
    else:
        smoothed_std = position_std

    smoothed_std = np.maximum(smoothed_std, 1e-8)
    return smoothed_std

def apply_reweighting(grads, reweighting_factors):
    """Apply re-weighting to gradient scores."""
    print("Applying re-weighting...")

    # grads: (n_genes, seq_len, 4) or (n_genes, seq_len, 4, n_tissues)
    if grads.ndim == 4:
        factors = reweighting_factors[None, :, None, None]
    else:
        factors = reweighting_factors[None, :, None]

    reweighted_grads = grads / factors
    return reweighted_grads

def save_preprocessed_data(grads, seqs, gene_ids, output_file, tissue_idx, use_residual, gaussian_sigma):
    """Save preprocessed data to HDF5 file in TF-MoDISco format."""
    print(f"Saving preprocessed data to {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        # Process gradients to correct shape
  
        if len(grads.shape) == 4:  # (n_genes, seq_len, 4, n_targets)
            if grads.shape[3] == 1:
                grads_2d = grads[:, :, :, 0]  # Shape: (n_genes, seq_len, 4)
            else:
                grads_2d = np.mean(grads, axis=3)  # Average across targets
        else: # (n_genes, seq_len, 4)
            grads_2d = grads
        
        # TF-MoDISco expects specific format based on source code analysis:
        # Format 1 (preferred): hyp_scores and input_seqs

        # hyp_scores: (n_samples, seq_len, 4) - attribution scores
        # input_seqs: (n_samples, seq_len, 4) - one-hot sequences
        
        print(f"  Saving in TF-MoDISco Format : hyp_scores + input_seqs")
        print(f"  Final gradients shape: {grads_2d.shape}")
        print(f"  Final sequences shape: {seqs.shape}")
        
        f.create_dataset('hyp_scores', data=grads_2d.astype(np.float32), compression='gzip')
        f.create_dataset('input_seqs', data=seqs.astype(np.float32), compression='gzip')
        
        # Also save gene IDs if available
        if gene_ids is not None:
            f.create_dataset('gene', data=gene_ids)
        
        # Save metadata
        f.attrs['tissue_index'] = tissue_idx
        f.attrs['format'] = 'TF-MoDISco Format : hyp_scores (attributions) + input_seqs (sequences)'
        
        # Set preprocessing description based on what was applied
        preprocessing_steps = []
        if use_residual:
            preprocessing_steps.append('residual_gradients')
        if gaussian_sigma is not None:
            preprocessing_steps.append('gaussian_reweighting')
        else:
            preprocessing_steps.append('std_reweighting')
        
        f.attrs['preprocessing'] = '_'.join(preprocessing_steps) if preprocessing_steps else 'raw_gradients'
        f.attrs['residual_computation'] = use_residual
        f.attrs['gaussian_filtering'] = gaussian_sigma is not None
        if gaussian_sigma is not None:
            f.attrs['gaussian_sigma'] = gaussian_sigma
        f.attrs['description'] = 'Preprocessed gradients for TF-MoDISco analysis'
        
    print(f"  Saved TF-MoDISco compatible file:")
    print(f"    hyp_scores: {grads_2d.shape} (attribution scores)")
    print(f"    input_seqs: {seqs.shape} (one-hot sequences)")

def save_preprocessed_data_to_group(grads, seqs, gene_ids, h5_handle, group_name, tissue_idx, use_residual, gaussian_sigma):
    """Save preprocessed data to an HDF5 group (for combined multi-tissue file)."""
    print(f"Saving tissue {tissue_idx} to group '{group_name}'...")

    grp = h5_handle.create_group(group_name)

    # Process gradients to correct shape
    if len(grads.shape) == 4:  # (n_genes, seq_len, 4, n_targets)
        if grads.shape[3] == 1:
            grads_2d = grads[:, :, :, 0]
        else:
            grads_2d = np.mean(grads, axis=3)
    else:
        grads_2d = grads

    grp.create_dataset('hyp_scores', data=grads_2d.astype(np.float32), compression='gzip')
    grp.create_dataset('input_seqs', data=seqs.astype(np.float32), compression='gzip')

    if gene_ids is not None:
        grp.create_dataset('gene', data=gene_ids)

    grp.attrs['tissue_index'] = tissue_idx
    grp.attrs['format'] = 'TF-MoDISco Format : hyp_scores (attributions) + input_seqs (sequences)'

    preprocessing_steps = []
    if use_residual:
        preprocessing_steps.append('residual_gradients')
    if gaussian_sigma is not None:
        preprocessing_steps.append('gaussian_reweighting')
    else:
        preprocessing_steps.append('std_reweighting')

    grp.attrs['preprocessing'] = '_'.join(preprocessing_steps) if preprocessing_steps else 'raw_gradients'
    grp.attrs['residual_computation'] = use_residual
    grp.attrs['gaussian_filtering'] = gaussian_sigma is not None
    if gaussian_sigma is not None:
        grp.attrs['gaussian_sigma'] = gaussian_sigma
    grp.attrs['description'] = 'Preprocessed gradients for TF-MoDISco analysis'

    print(f"  Saved group '{group_name}': hyp_scores {grads_2d.shape}, input_seqs {seqs.shape}")

def process_tissue_efficient(tissue_idx, grad_dir, center_size, gaussian_sigma, gaussian_truncate, use_residual, out_dir):
    """Process a single tissue from its own H5 file that contains all tissues for its top genes."""
    print(f"\nProcessing tissue {tissue_idx} from per-tissue H5...")

    grad_file = os.path.join(grad_dir, f"tissue_{tissue_idx}.h5")
    grads_all_tissues, seqs, gene_ids = load_gradient_data(grad_file, center_size)
    # grads_all_tissues: (n_genes, seq_len, 4, n_tissues)

    # Extract current tissue gradients
    tissue_grads = grads_all_tissues[:, :, :, tissue_idx]  # (n_genes, seq_len, 4)

    if use_residual:
        print("  Computing residual gradients (tissue - avg(others))")
        other = np.delete(grads_all_tissues, tissue_idx, axis=3)  # (n_genes, seq_len, 4, n_tissues-1)
        other_mean = np.mean(other, axis=3)  # (n_genes, seq_len, 4)
        final_grads = tissue_grads - other_mean
    else:
        final_grads = tissue_grads

    # Zero-mean across nucleotides at each position (per sample, per position)
    zero_mean_tissue_grads = tissue_grads - tissue_grads.mean(axis=2, keepdims=True)
    zero_mean_final_grads = final_grads - final_grads.mean(axis=2, keepdims=True)

    # Compute reweighting factors on zero-mean tissue gradients
    reweighting_factors = compute_reweighting_factors(zero_mean_tissue_grads, gaussian_sigma, gaussian_truncate)

    # Apply reweighting to zero-mean final gradients
    final_grads = apply_reweighting(zero_mean_final_grads, reweighting_factors)  # keeps (n_genes, seq_len, 4)

    # Save
    output_file = os.path.join(out_dir, f"tissue_{tissue_idx}_preprocessed.h5")
    save_preprocessed_data(final_grads, seqs, gene_ids, output_file, tissue_idx, use_residual, gaussian_sigma)
    return output_file

def process_tissue_compute_only(tissue_idx, grad_dir, center_size, gaussian_sigma, gaussian_truncate, use_residual):
    """Process a single tissue and return arrays without saving to an individual file."""
    print(f"\nProcessing tissue {tissue_idx} (compute-only)...")

    grad_file = os.path.join(grad_dir, f"tissue_{tissue_idx}.h5")
    grads_all_tissues, seqs, gene_ids = load_gradient_data(grad_file, center_size)
    # grads_all_tissues: (n_genes, seq_len, 4, n_tissues)

    tissue_grads = grads_all_tissues[:, :, :, tissue_idx]  # (n_genes, seq_len, 4)

    if use_residual:
        print("  Computing residual gradients (tissue - avg(others))")
        other = np.delete(grads_all_tissues, tissue_idx, axis=3)
        other_mean = np.mean(other, axis=3)
        final_grads = tissue_grads - other_mean
    else:
        final_grads = tissue_grads

    zero_mean_tissue_grads = tissue_grads - tissue_grads.mean(axis=2, keepdims=True)
    zero_mean_final_grads = final_grads - final_grads.mean(axis=2, keepdims=True)

    reweighting_factors = compute_reweighting_factors(zero_mean_tissue_grads, gaussian_sigma, gaussian_truncate)

    final_grads = apply_reweighting(zero_mean_final_grads, reweighting_factors)

    return final_grads, seqs, gene_ids

def main():
    """Main function."""
    args = parse_args()
    
    # Parse tissue indices
    try:
        tissue_indices = [int(x.strip()) for x in args.tissue_indices.split(',')]
        print(f"Processing tissues: {tissue_indices}")
    except ValueError:
        print("Error: Invalid tissue indices. Please provide comma-separated integers.")
        sys.exit(1)
    
    # Check input directory
    if not os.path.exists(args.grad_dir):
        print(f"Error: Gradient directory not found: {args.grad_dir}")
        sys.exit(1)
    
    os.makedirs(args.out_dir, exist_ok=True)

    combined_out_file = os.path.join(args.out_dir, 'all_tissues_preprocessed.h5')
    print(f"\nCreating combined output file: {combined_out_file}")
    with h5py.File(combined_out_file, 'w') as h5f:
        h5f.attrs['description'] = 'Combined preprocessed gradients for TF-MoDISco analysis (multiple tissues)'
        h5f.attrs['tissue_indices_order'] = ','.join(str(x) for x in tissue_indices)
        h5f.attrs['residual_computation'] = args.residual
        h5f.attrs['gaussian_filtering'] = args.gaussian_sigma is not None
        if args.gaussian_sigma is not None:
            h5f.attrs['gaussian_sigma'] = args.gaussian_sigma
            if args.gaussian_truncate is not None:
                h5f.attrs['gaussian_truncate'] = args.gaussian_truncate

        for tissue_idx in tissue_indices:
            try:
                print(f"\n{'='*50}")
                final_grads, seqs, gene_ids = process_tissue_compute_only(
                    tissue_idx,
                    args.grad_dir,
                    args.center_size,
                    args.gaussian_sigma,
                    args.gaussian_truncate,
                    args.residual,
                )
                group_name = f"tissue_{tissue_idx}"
                save_preprocessed_data_to_group(
                    final_grads,
                    seqs,
                    gene_ids,
                    h5f,
                    group_name,
                    tissue_idx,
                    args.residual,
                    args.gaussian_sigma,
                )
                print(f"Completed tissue {tissue_idx} -> group '{group_name}'")
            except Exception as e:
                print(f"Error processing tissue {tissue_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\nPreprocessing completed successfully!")
    print(f"Combined output file: {combined_out_file}")
    print(f"Processed {len(tissue_indices)} tissues")
 

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv += [
        "--grad_dir", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/5tissues_atg_grads",
        "--tissue_indices", "18,0,22,19,20",
        "-o", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/tmp/modisco_tmp_preprocessed",
        "--residual",
        "--gaussian_sigma", "1280.0",
        "--gaussian_truncate", "2.0"
        ]
    main()