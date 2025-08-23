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
    parser = argparse.ArgumentParser(
        description="Preprocess gradients for TF-MoDISco analysis (concat all tissues into one H5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--grad_dir",
        required=True,
        help="Directory containing per-tissue H5 files: tissue_{idx}.h5 (with datasets: grads, seqs, [gene])"
    )
    parser.add_argument(
        "--tissue_indices",
        required=True,
        help="Comma-separated tissue indices to process (e.g., 18,0,22,19,20). Order defines the concat order."
    )
    parser.add_argument(
        "-o", "--out",
        default="modisco_preprocessed_all",
        help="Output file path OR directory. Extension will be added based on --out_format"
    )
    parser.add_argument(
        "--out_format",
        choices=["h5", "npy", "npz"],
        default="h5",
        help="Output format: h5 (HDF5), npy (numpy array), npz (compressed numpy). Default: h5"
    )
    parser.add_argument(
        "--no_gene_info",
        action="store_true",
        help="Exclude gene information from output (useful for npy format)"
    )
    parser.add_argument(
        "--split_by_tissues",
        action="store_true",
        help="Generate separate files for each tissue instead of merging into one file"
    )
    parser.add_argument(
        "--center_size",
        type=int,
        default=32768,
        help="Size of centered region to extract (in bp)"
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=float,
        default=None,
        help="Std dev for Gaussian filter (for reweighting factors). If None, no Gaussian smoothing."
    )
    parser.add_argument(
        "--gaussian_truncate",
        type=float,
        default=None,
        help="Truncate for Gaussian filter (only used if --gaussian_sigma is set; default 4.0)"
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Compute residual gradients (tissue - avg(others)) using the per-file tensor."
    )
    return parser.parse_args()

def resolve_out_path(out_arg, out_format, tissue_idx=None, split_by_tissues=False):
    """Resolve output path based on format."""
    p = Path(out_arg)
    
    # Map format to extension
    ext_map = {"h5": ".h5", "npy": ".npy", "npz": ".npz"}
    target_ext = ext_map[out_format]
    
    # If already has correct extension, use as-is
    if p.suffix.lower() == target_ext:
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    
    # If has different extension, replace it
    if p.suffix:
        p = p.with_suffix(target_ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    
    # No extension - treat as directory or base name
    if p.is_dir() or str(p).endswith('/'):
        # It's a directory
        Path(p).mkdir(parents=True, exist_ok=True)
        if split_by_tissues and tissue_idx is not None:
            return str(Path(p) / f"modisco_preprocessed_tissue_{tissue_idx}{target_ext}")
        else:
            return str(Path(p) / f"modisco_preprocessed_all{target_ext}")
    else:
        # No extension - treat as base filename
        if split_by_tissues and tissue_idx is not None:
            # Add tissue index to filename
            tissue_suffix = f"_tissue_{tissue_idx}"
            p = p.with_name(f"{p.name}{tissue_suffix}")
        p = p.with_suffix(target_ext)
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

def load_gradient_data(grad_file, center_size=None):
    print(f"Loading gradients from {grad_file}...")
    with h5py.File(grad_file, 'r') as f:
        grads_shape = f['grads'].shape
        seqs_shape = f['seqs'].shape
        print(f"  File gradients shape: {grads_shape}")
        print(f"  File sequences shape: {seqs_shape}")

        grads_dtype = f['grads'].dtype
        target_dtype = np.float32 if grads_dtype == np.float16 else grads_dtype

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

def compute_reweighting_factors(grads, gaussian_sigma, gaussian_truncate):
    print("Computing re-weighting factors...")
    print(f"  Input gradients shape: {grads.shape}")
    print("  Computing position-wise std across nucleotides...")

    chunk_size = min(1000, grads.shape[0])
    position_stds = []
    for i in range(0, grads.shape[0], chunk_size):
        end_idx = min(i + chunk_size, grads.shape[0])
        chunk = grads[i:end_idx]           # (chunk_size, seq_len, 4)
        chunk_std = np.std(chunk, axis=2)  # (chunk_size, seq_len)
        chunk_position_std = np.mean(chunk_std, axis=0)  # (seq_len,)
        position_stds.append(chunk_position_std)

    position_std = np.mean(position_stds, axis=0)       # (seq_len,)
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
    print("Applying re-weighting...")
    if grads.ndim == 4:
        factors = reweighting_factors[None, :, None, None]
    else:
        factors = reweighting_factors[None, :, None]
    return grads / factors

def process_one_tissue(tissue_idx, grad_dir, center_size, gaussian_sigma, gaussian_truncate, use_residual):
    print(f"\nProcessing tissue {tissue_idx} ...")
    grad_file = os.path.join(grad_dir, f"tissue_{tissue_idx}.h5")
    grads_all_tissues, seqs, gene_ids = load_gradient_data(grad_file, center_size)
    # grads_all_tissues: (n_genes, seq_len, 4, n_tissues)

    tissue_grads = grads_all_tissues[:, :, :, tissue_idx]  # (n_genes, seq_len, 4)

    if use_residual:
        print("  Computing residual gradients (tissue - mean(others))")
        other = np.delete(grads_all_tissues, tissue_idx, axis=3)  # (n_genes, seq_len, 4, n_tissues-1)
        other_mean = np.mean(other, axis=3)                       # (n_genes, seq_len, 4)
        final_grads = tissue_grads - other_mean
    else:
        final_grads = tissue_grads

    # Zero-mean across nucleotides per position (both for factor computation和最终打分)
    zero_mean_tissue = tissue_grads - tissue_grads.mean(axis=2, keepdims=True)
    zero_mean_final  = final_grads  - final_grads.mean(axis=2, keepdims=True)

    # Reweighting factors computed on zero-mean tissue (与原逻辑一致)
    rw = compute_reweighting_factors(zero_mean_tissue, gaussian_sigma, gaussian_truncate)
    final_grads = apply_reweighting(zero_mean_final, rw).astype(np.float32)

    return final_grads, seqs, gene_ids  # shapes: (n_genes, L, 4), (n_genes, L, 4), (n_genes,) or None

def ensure_datasets(h5f, seq_len, compression="gzip"):
    """Create resizable datasets if not exist."""
    if 'hyp_scores' not in h5f:
        h5f.create_dataset(
            'hyp_scores',
            shape=(0, seq_len, 4),
            maxshape=(None, seq_len, 4),
            dtype=np.float32,
            compression=compression
        )
    if 'input_seqs' not in h5f:
        h5f.create_dataset(
            'input_seqs',
            shape=(0, seq_len, 4),
            maxshape=(None, seq_len, 4),
            dtype=np.float32,
            compression=compression
        )
    if 'tissue_index' not in h5f:
        h5f.create_dataset(
            'tissue_index',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            compression=compression
        )
    if 'gene' not in h5f:
        # 可变长 UTF-8 字符串
        dt = h5py.string_dtype(encoding='utf-8')
        h5f.create_dataset(
            'gene',
            shape=(0,),
            maxshape=(None,),
            dtype=dt,
            compression=compression
        )

def append_block(h5f, grads_block, seqs_block, gene_block, tissue_idx):
    """Append a block to resizable datasets."""
    n_new, L, C = grads_block.shape
    # Sanity
    assert C == 4, "Channel dimension must be 4"

    # Current sizes
    n_curr = h5f['hyp_scores'].shape[0]

    # Resize
    h5f['hyp_scores'].resize((n_curr + n_new, L, C))
    h5f['input_seqs'].resize((n_curr + n_new, L, C))
    h5f['tissue_index'].resize((n_curr + n_new,))
    h5f['gene'].resize((n_curr + n_new,))

    # Write
    h5f['hyp_scores'][n_curr:n_curr+n_new, :, :] = grads_block
    h5f['input_seqs'][n_curr:n_curr+n_new, :, :] = seqs_block
    h5f['tissue_index'][n_curr:n_curr+n_new] = tissue_idx

    if gene_block is None:
        h5f['gene'][n_curr:n_curr+n_new] = [""] * n_new
    else:
        if isinstance(gene_block, np.ndarray) and gene_block.dtype.kind in ('S', 'U', 'O'):
            h5f['gene'][n_curr:n_curr+n_new] = gene_block.astype(str)
        else:
            h5f['gene'][n_curr:n_curr+n_new] = [str(x) for x in gene_block]

def save_as_npy_npz(out_path, out_format, all_grads, all_seqs, all_tissue_indices, all_genes, 
                    tissue_indices, args):
    """Save data in npy or npz format."""
    print(f"Saving data in {out_format.upper()} format...")
    
    # Convert lists to numpy arrays
    hyp_scores = np.concatenate(all_grads, axis=0).astype(np.float32)
    input_seqs = np.concatenate(all_seqs, axis=0).astype(np.float32)
    tissue_index = np.concatenate(all_tissue_indices, axis=0).astype(np.int32)
    
    print(f"  Final shapes:")
    print(f"    hyp_scores: {hyp_scores.shape}")
    print(f"    input_seqs: {input_seqs.shape}")
    print(f"    tissue_index: {tissue_index.shape}")
    
    if out_format == "npy":
        # For npy format, save as separate files
        base_path = Path(out_path).with_suffix('')
        
        np.save(f"{base_path}_hyp_scores.npy", hyp_scores)
        np.save(f"{base_path}_input_seqs.npy", input_seqs)
        np.save(f"{base_path}_tissue_index.npy", tissue_index)
        
        print(f"  Saved files:")
        print(f"    {base_path}_hyp_scores.npy")
        print(f"    {base_path}_input_seqs.npy")
        print(f"    {base_path}_tissue_index.npy")
        
        if not args.no_gene_info and all_genes:
            gene_array = np.concatenate(all_genes, axis=0)
            np.save(f"{base_path}_gene.npy", gene_array)
            print(f"    {base_path}_gene.npy")
            
    elif out_format == "npz":
        # For npz format, save as compressed archive
        save_dict = {
            'hyp_scores': hyp_scores,
            'input_seqs': input_seqs,
            'tissue_index': tissue_index,
            'tissue_indices_order': np.array(tissue_indices, dtype=np.int32),
            'residual_computation': bool(args.residual),
            'gaussian_filtering': args.gaussian_sigma is not None,
            'center_size': int(hyp_scores.shape[1])
        }
        
        if args.gaussian_sigma is not None:
            save_dict['gaussian_sigma'] = float(args.gaussian_sigma)
            if args.gaussian_truncate is not None:
                save_dict['gaussian_truncate'] = float(args.gaussian_truncate)
        
        if not args.no_gene_info and all_genes:
            gene_array = np.concatenate(all_genes, axis=0)
            save_dict['gene'] = gene_array
            print(f"    gene: {gene_array.shape}")
        
        np.savez_compressed(out_path, **save_dict)
        print(f"  Saved file: {out_path}")

def main():
    args = parse_args()

    try:
        tissue_indices = [int(x.strip()) for x in args.tissue_indices.split(',')]
        print(f"Processing tissues in order: {tissue_indices}")
    except ValueError:
        print("Error: Invalid tissue indices. Please provide comma-separated integers.")
        sys.exit(1)

    if not os.path.exists(args.grad_dir):
        print(f"Error: Gradient directory not found: {args.grad_dir}")
        sys.exit(1)

    if args.split_by_tissues:
        print("Mode: Generate separate files for each tissue")
    else:
        print("Mode: Merge all tissues into one file")
        out_path = resolve_out_path(args.out, args.out_format)
        print(f"Output will be saved to: {out_path}")

    probe_file = os.path.join(args.grad_dir, f"tissue_{tissue_indices[0]}.h5")
    if not os.path.exists(probe_file):
        print(f"Error: Missing file for tissue {tissue_indices[0]}: {probe_file}")
        sys.exit(1)

    with h5py.File(probe_file, 'r') as f0:
        full_L = f0['grads'].shape[1]
    L = min(args.center_size, full_L) if args.center_size is not None else full_L
    print(f"Sequence length to save: {L}")

    if args.split_by_tissues:
        # Generate separate files for each tissue
        total_samples = 0
        for tissue_idx in tissue_indices:
            print("\n" + "="*60)
            print(f"Processing tissue {tissue_idx}...")
            
            try:
                grads_block, seqs_block, gene_block = process_one_tissue(
                    tissue_idx=tissue_idx,
                    grad_dir=args.grad_dir,
                    center_size=args.center_size,
                    gaussian_sigma=args.gaussian_sigma,
                    gaussian_truncate=args.gaussian_truncate,
                    use_residual=args.residual,
                )
                
                if grads_block.shape[1] != L:
                    seq_len = grads_block.shape[1]
                    if L <= seq_len:
                        start = (seq_len - L) // 2
                        end = start + L
                        grads_block = grads_block[:, start:end, :]
                        seqs_block  = seqs_block[:,  start:end, :]
                    else:
                        raise RuntimeError(f"Computed block length {seq_len} < target L {L}")

                # Generate output path for this tissue
                tissue_out_path = resolve_out_path(args.out, args.out_format, tissue_idx, True)
                
                # Save tissue-specific data
                if args.out_format == "h5":
                    # H5 format for single tissue
                    with h5py.File(tissue_out_path, 'w') as out_h5:
                        # Create datasets
                        out_h5.create_dataset('hyp_scores', data=grads_block, compression="gzip")
                        out_h5.create_dataset('input_seqs', data=seqs_block, compression="gzip")
                        out_h5.create_dataset('tissue_index', data=np.array([tissue_idx], dtype=np.int32), compression="gzip")
                        
                        if not args.no_gene_info and gene_block is not None:
                            if isinstance(gene_block, np.ndarray) and gene_block.dtype.kind in ('S', 'U', 'O'):
                                out_h5.create_dataset('gene', data=gene_block.astype(str), compression="gzip")
                            else:
                                out_h5.create_dataset('gene', data=np.array([str(x) for x in gene_block]), compression="gzip")
                        
                        # Add metadata
                        out_h5.attrs['format'] = 'TF-MoDISco Format : hyp_scores (attributions) + input_seqs (sequences)'
                        out_h5.attrs['description'] = f'Preprocessed gradients for TF-MoDISco analysis (tissue {tissue_idx})'
                        out_h5.attrs['tissue_index'] = int(tissue_idx)
                        out_h5.attrs['residual_computation'] = bool(args.residual)
                        out_h5.attrs['gaussian_filtering'] = args.gaussian_sigma is not None
                        if args.gaussian_sigma is not None:
                            out_h5.attrs['gaussian_sigma'] = float(args.gaussian_sigma)
                            if args.gaussian_truncate is not None:
                                out_h5.attrs['gaussian_truncate'] = float(args.gaussian_truncate)
                        out_h5.attrs['center_size'] = int(L)
                        
                        print(f"  Saved H5 file: {tissue_out_path}")
                        print(f"  Datasets:")
                        print(f"    hyp_scores: {out_h5['hyp_scores'].shape}")
                        print(f"    input_seqs: {out_h5['input_seqs'].shape}")
                        print(f"    tissue_index: {out_h5['tissue_index'].shape}")
                        if 'gene' in out_h5:
                            print(f"    gene: {out_h5['gene'].shape}")
                
                else:
                    # NPY/NPZ format for single tissue
                    if args.out_format == "npy":
                        # Save as separate NPY files
                        base_path = Path(tissue_out_path).with_suffix('')
                        np.save(f"{base_path}_hyp_scores.npy", grads_block)
                        np.save(f"{base_path}_input_seqs.npy", seqs_block)
                        np.save(f"{base_path}_tissue_index.npy", np.array([tissue_idx], dtype=np.int32))
                        
                        if not args.no_gene_info and gene_block is not None:
                            if isinstance(gene_block, np.ndarray) and gene_block.dtype.kind in ('S', 'U', 'O'):
                                np.save(f"{base_path}_gene.npy", gene_block.astype(str))
                            else:
                                np.save(f"{base_path}_gene.npy", np.array([str(x) for x in gene_block]))
                        
                        print(f"  Saved NPY files:")
                        print(f"    {base_path}_hyp_scores.npy")
                        print(f"    {base_path}_input_seqs.npy")
                        print(f"    {base_path}_tissue_index.npy")
                        if not args.no_gene_info and gene_block is not None:
                            print(f"    {base_path}_gene.npy")
                    
                    elif args.out_format == "npz":
                        # Save as NPZ file
                        save_dict = {
                            'hyp_scores': grads_block,
                            'input_seqs': seqs_block,
                            'tissue_index': np.array([tissue_idx], dtype=np.int32),
                            'residual_computation': bool(args.residual),
                            'gaussian_filtering': args.gaussian_sigma is not None,
                            'center_size': int(L)
                        }
                        
                        if args.gaussian_sigma is not None:
                            save_dict['gaussian_sigma'] = float(args.gaussian_sigma)
                            if args.gaussian_truncate is not None:
                                save_dict['gaussian_truncate'] = float(args.gaussian_truncate)
                        
                        if not args.no_gene_info and gene_block is not None:
                            if isinstance(gene_block, np.ndarray) and gene_block.dtype.kind in ('S', 'U', 'O'):
                                save_dict['gene'] = gene_block.astype(str)
                            else:
                                save_dict['gene'] = np.array([str(x) for x in gene_block])
                        
                        np.savez_compressed(tissue_out_path, **save_dict)
                        print(f"  Saved NPZ file: {tissue_out_path}")
                        print(f"    hyp_scores: {grads_block.shape}")
                        print(f"    input_seqs: {seqs_block.shape}")
                        print(f"    tissue_index: (1,)")
                        if not args.no_gene_info and gene_block is not None:
                            print(f"    gene: {gene_block.shape}")
                
                total_samples += grads_block.shape[0]
                print(f"  Tissue {tissue_idx}: {grads_block.shape[0]} samples")
                del grads_block, seqs_block, gene_block
                
            except Exception as e:
                print(f"Error processing tissue {tissue_idx}: {e}")
                import traceback; traceback.print_exc()
                continue
        
        print(f"\nAll tissues processed separately!")
        print(f"  Total samples across all tissues: {total_samples}")
        print(f"  Generated {len(tissue_indices)} tissue-specific files")
    
    else:
        # Original logic: merge all tissues into one file
        if args.out_format == "h5":
            # H5 format: write data incrementally
            with h5py.File(out_path, 'w') as out_h5:
                ensure_datasets(out_h5, seq_len=L, compression="gzip")

                out_h5.attrs['format'] = 'TF-MoDISco Format : hyp_scores (attributions) + input_seqs (sequences)'
                out_h5.attrs['description'] = 'Preprocessed gradients for TF-MoDISco analysis (all tissues concatenated)'
                out_h5.attrs['tissue_indices_order'] = np.array(tissue_indices, dtype=np.int32)
                out_h5.attrs['residual_computation'] = bool(args.residual)
                out_h5.attrs['gaussian_filtering'] = args.gaussian_sigma is not None
                if args.gaussian_sigma is not None:
                    out_h5.attrs['gaussian_sigma'] = float(args.gaussian_sigma)
                    if args.gaussian_truncate is not None:
                        out_h5.attrs['gaussian_truncate'] = float(args.gaussian_truncate)
                out_h5.attrs['center_size'] = int(L)

                total = 0
                for tissue_idx in tissue_indices:
                    print("\n" + "="*60)
                    try:
                        grads_block, seqs_block, gene_block = process_one_tissue(
                            tissue_idx=tissue_idx,
                            grad_dir=args.grad_dir,
                            center_size=args.center_size,
                            gaussian_sigma=args.gaussian_sigma,
                            gaussian_truncate=args.gaussian_truncate,
                            use_residual=args.residual,
                        )
                        if grads_block.shape[1] != L:
                            seq_len = grads_block.shape[1]
                            if L <= seq_len:
                                start = (seq_len - L) // 2
                                end = start + L
                                grads_block = grads_block[:, start:end, :]
                                seqs_block  = seqs_block[:,  start:end, :]
                            else:
                                raise RuntimeError(f"Computed block length {seq_len} < target L {L}")

                        append_block(out_h5, grads_block, seqs_block, gene_block, tissue_idx)
                        total += grads_block.shape[0]
                        print(f"Appended tissue {tissue_idx}: +{grads_block.shape[0]} samples (total={total})")
                        del grads_block, seqs_block, gene_block
                    except Exception as e:
                        print(f"Error processing tissue {tissue_idx}: {e}")
                        import traceback; traceback.print_exc()
                        continue

                print("\nAll tissues processed!")
                print(f"  Output file: {out_path}")
                print(f"  Total samples: {total}")
                print(f"  Datasets:")
                print(f"    hyp_scores: {out_h5['hyp_scores'].shape}")
                print(f"    input_seqs: {out_h5['input_seqs'].shape}")
                print(f"    tissue_index: {out_h5['tissue_index'].shape}")
                print(f"    gene: {out_h5['gene'].shape}")
        
        else:
            # NPY/NPZ format: collect all data in memory, then save
            all_grads = []
            all_seqs = []
            all_tissue_indices = []
            all_genes = []
            
            total = 0
            for tissue_idx in tissue_indices:
                print("\n" + "="*60)
                try:
                    grads_block, seqs_block, gene_block = process_one_tissue(
                        tissue_idx=tissue_idx,
                        grad_dir=args.grad_dir,
                        center_size=args.center_size,
                        gaussian_sigma=args.gaussian_sigma,
                        gaussian_truncate=args.gaussian_truncate,
                        use_residual=args.residual,
                    )
                    if grads_block.shape[1] != L:
                        seq_len = grads_block.shape[1]
                        if L <= seq_len:
                            start = (seq_len - L) // 2
                            end = start + L
                            grads_block = grads_block[:, start:end, :]
                            seqs_block  = seqs_block[:,  start:end, :]
                        else:
                            raise RuntimeError(f"Computed block length {seq_len} < target L {L}")

                    # Collect data for later saving
                    all_grads.append(grads_block)
                    all_seqs.append(seqs_block)
                    
                    # Create tissue index array for this block
                    tissue_idx_array = np.full(grads_block.shape[0], tissue_idx, dtype=np.int32)
                    all_tissue_indices.append(tissue_idx_array)
                    
                    # Handle gene information
                    if not args.no_gene_info and gene_block is not None:
                        if isinstance(gene_block, np.ndarray) and gene_block.dtype.kind in ('S', 'U', 'O'):
                            all_genes.append(gene_block.astype(str))
                        else:
                            all_genes.append(gene_block.astype(str))
                    elif not args.no_gene_info:
                        # Create empty gene array if no gene info available
                        all_genes.append(np.array([""] * grads_block.shape[0]))
                    
                    total += grads_block.shape[0]
                    print(f"Collected tissue {tissue_idx}: +{grads_block.shape[0]} samples (total={total})")
                    del grads_block, seqs_block, gene_block

                except Exception as e:
                    print(f"Error processing tissue {tissue_idx}: {e}")
                    import traceback; traceback.print_exc()
                    continue
            
            # Save collected data
            if all_grads:
                save_as_npy_npz(out_path, args.out_format, all_grads, all_seqs, 
                               all_tissue_indices, all_genes if not args.no_gene_info else None,
                               tissue_indices, args)
                print(f"\nAll tissues processed! Total samples: {total}")
            else:
                print("No data was successfully processed!")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += [
            "--grad_dir", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/5tissues_atg_grads",
            "--tissue_indices", "18,0,22,19,20",
            "-o", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/tmp/modisco_tmp_preprocessed_all",
            "--out_format", "h5",
            "--split_by_tissues",
            "--residual",
            "--gaussian_sigma", "1280.0",
            "--gaussian_truncate", "2.0"
        ]
    main()