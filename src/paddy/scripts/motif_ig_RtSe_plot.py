#!/usr/bin/env python3
"""
Motif Visualization Script
Visualizes gradient saliency for genes and their motif regions
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Set Arial font globally and non-interactive backend for multiprocessing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

# Helper function to draw a letter at a given position
def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):
    """Draw DNA letter at specified position with given scale and color"""
    
    # Define letter heights and colors
    fp = FontProperties(family="Arial", weight="bold")
    globscale = 1.35
    LETTERS = {
        "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
        "UP": TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
        "DN": TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp),
        "(": TextPath((-0.25, 0), "(", size=1, prop=fp),
        ".": TextPath((-0.125, 0), "-", size=1, prop=fp),
        ")": TextPath((-0.1, 0), ")", size=1, prop=fp)
    }
    COLOR_SCHEME = {
        'G': 'orange',
        'A': 'green',
        'C': 'blue',
        'T': 'red',
        'UP': 'green',
        'DN': 'red',
        '(': 'black',
        '.': 'black',
        ')': 'black'
    }

    text = LETTERS[letter]

    # Choose color
    chosen_color = COLOR_SCHEME[letter]
    if color is not None:
        chosen_color = color

    # Draw letter onto axis
    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x, y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax is not None:
        ax.add_artist(p)
    
    return p

def plot_seq_scores(importance_scores, figsize=(16, 2), plot_y_ticks=True, 
                   y_min=None, y_max=None, save_figs=False, fig_name="default", 
                   center_marker=False, remove_spines=False, title=None, 
                   mode='max-grad', ref_seq=None, symmetric_y=True, percentile_clip=None):
    """Plot sequence logo from importance scores
    
    Args:
        importance_scores: Shape (length, 4) or (4, length) - attribution scores for ACGT
        mode: 'true-seq' or 'max-grad'
            - 'true-seq': Display actual sequence with its attribution values
            - 'max-grad': Display nucleotide with max |attribution| at each position
        ref_seq: Reference sequence string (required for 'true-seq' mode)
        symmetric_y: If True, use symmetric y-axis around 0
        percentile_clip: If set, clip y values at percentile (e.g., 95 clips outliers)
    """
    
    importance_scores = importance_scores.T if len(importance_scores.shape) == 2 and importance_scores.shape[0] != 4 else importance_scores
    fig = plt.figure(figsize=figsize)
    
    # Determine sequence and scores based on mode
    if mode == 'true-seq':
        if ref_seq is None:
            raise ValueError("ref_seq must be provided for 'true-seq' mode")
        display_seq = ref_seq.upper()
        scores = []
        nt_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for i, nt in enumerate(display_seq):
            if nt in nt_to_idx:
                scores.append(importance_scores[nt_to_idx[nt], i])
            else:
                scores.append(0.0)  # Unknown nucleotide
    elif mode == 'max-grad':
        display_seq = ""
        scores = []
        idx_to_nt = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        for j in range(importance_scores.shape[1]):
            argmax_nt = np.argmax(np.abs(importance_scores[:, j]))
            display_seq += idx_to_nt[argmax_nt]
            scores.append(importance_scores[argmax_nt, j])
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'true-seq' or 'max-grad'")

    scores = np.array(scores)

    ax = plt.gca()
    if remove_spines:
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)

    if center_marker:
        center_x = len(display_seq) / 2
        plt.axvline(x=center_x, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Draw letters
    for i in range(len(display_seq)):
        mutability_score = scores[i]
        dna_letter_at(display_seq[i], i + 0.5, 0, mutability_score, ax, color=None)
    
    plt.sca(ax)
    plt.xticks([], [])
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    plt.xlim((0, len(display_seq)))
    
    if title:
        plt.title(title, fontsize=10)
    
    if plot_y_ticks:
        plt.yticks(fontsize=10)
    else:
        plt.yticks([], [])
    
    # Set axis limits with improved logic
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(y_min)
    else:
        # Apply percentile clipping if requested
        if percentile_clip is not None:
            score_min = np.percentile(scores, 100 - percentile_clip)
            score_max = np.percentile(scores, percentile_clip)
        else:
            score_min = np.min(scores)
            score_max = np.max(scores)
        
        # Symmetric y-axis
        if symmetric_y:
            max_abs = max(abs(score_min), abs(score_max))
            plt.ylim(-max_abs * 1.1, max_abs * 1.1)
        else:
            plt.ylim(score_min - 0.1 * abs(score_max - score_min),
                    score_max + 0.1 * abs(score_max - score_min))
    
    plt.axhline(y=0., color='black', linestyle='-', linewidth=1)
    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_name + ".png", transparent=True, dpi=300)
        plt.savefig(fig_name + ".pdf", transparent=True, dpi=300)

    return fig

def plot_seq_scores_on_axis(importance_scores, ax=None, plot_y_ticks=True, 
                           y_min=None, y_max=None, center_marker=False, 
                           remove_spines=False, title=None, mode='max-grad', 
                           ref_seq=None, symmetric_y=True, percentile_clip=None):
    """Plot sequence logo from importance scores on specified axis
    
    Args:
        importance_scores: Shape (length, 4) or (4, length) - attribution scores for ACGT
        ax: Matplotlib axis to plot on (if None, uses current axis)
        mode: 'true-seq' or 'max-grad'
            - 'true-seq': Display actual sequence with its attribution values
            - 'max-grad': Display nucleotide with max |attribution| at each position
        ref_seq: Reference sequence string (required for 'true-seq' mode)
        symmetric_y: If True, use symmetric y-axis around 0
        percentile_clip: If set, clip y values at percentile (e.g., 95 clips outliers)
    """
    
    # Ensure we have the right shape: (4, length) for nucleotides x positions
    if len(importance_scores.shape) == 2:
        if importance_scores.shape[0] != 4:
            importance_scores = importance_scores.T  # (length, 4) -> (4, length)
    
    # Determine sequence and scores based on mode
    if mode == 'true-seq':
        if ref_seq is None:
            raise ValueError("ref_seq must be provided for 'true-seq' mode")
        display_seq = ref_seq.upper()
        scores = []
        nt_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for i, nt in enumerate(display_seq):
            if nt in nt_to_idx:
                scores.append(importance_scores[nt_to_idx[nt], i])
            else:
                scores.append(0.0)  # Unknown nucleotide
    elif mode == 'max-grad':
        display_seq = ""
        scores = []
        idx_to_nt = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        for j in range(importance_scores.shape[1]):
            argmax_nt = np.argmax(np.abs(importance_scores[:, j]))
            display_seq += idx_to_nt[argmax_nt]
            scores.append(importance_scores[argmax_nt, j])
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'true-seq' or 'max-grad'")

    scores = np.array(scores)

    if ax is None:
        ax = plt.gca()
    
    if remove_spines:
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)

    # Center marker only for specific cases
    if center_marker:
        center_x = len(display_seq) / 2
        ax.axvline(x=center_x, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Draw letters
    for i in range(len(display_seq)):
        mutability_score = scores[i]
        dna_letter_at(display_seq[i], i + 0.5, 0, mutability_score, ax, color=None)
    
    ax.set_xticks([])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    ax.set_xlim((0, len(display_seq)))
    
    if title:
        ax.set_title(title, fontsize=10)
    
    if plot_y_ticks:
        ax.tick_params(axis='y', labelsize=10)
    else:
        ax.set_yticks([])
    
    # Set axis limits with improved logic
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    elif y_min is not None:
        ax.set_ylim(y_min)
    else:
        # Apply percentile clipping if requested
        if percentile_clip is not None:
            score_min = np.percentile(scores, 100 - percentile_clip)
            score_max = np.percentile(scores, percentile_clip)
        else:
            score_min = np.min(scores)
            score_max = np.max(scores)
        
        # Symmetric y-axis
        if symmetric_y:
            max_abs = max(abs(score_min), abs(score_max))
            ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)
        else:
            ax.set_ylim(score_min - 0.1 * abs(score_max - score_min),
                       score_max + 0.1 * abs(score_max - score_min))
    
    ax.axhline(y=0., color='black', linestyle='-', linewidth=0.5)

def load_data(h5_file='all_genes.h5', motif_json='motif_info.json', 
              fasta_file='all_genes.fa', tissue_dict_file='../23tissues_dict.json'):
    """Load all necessary data files
    
    Args:
        h5_file: Path to HDF5 file with integrated gradients data
        motif_json: Path to motif information JSON file
        fasta_file: Path to FASTA file with gene sequences
        tissue_dict_file: Path to tissue dictionary JSON file
    
    """
    print("Loading data...")
    print(f"  H5 file: {h5_file}")
    print(f"  Motif JSON: {motif_json}")
    print(f"  FASTA file: {fasta_file}")
    print(f"  Tissue dict: {tissue_dict_file}")
    
    # Load integrated gradients data
    with h5py.File(h5_file, 'r') as f:
        print("Datasets in HDF5 file:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}: {dataset.shape} {dataset.dtype}")
        
        fasta_headers = f['fasta_headers'][:]
        
        # Load igs dataset
        if 'igs' in f:
            ig_data = f['igs'][:]
        else:
            raise KeyError("'igs' not found in HDF5 file. Available keys: " + str(list(f.keys())))
        
        seqs = f['seqs'][:]
        
    # Load motif information
    with open(motif_json, 'r') as f:
        motif_info = json.load(f)
    
    # Load tissue dictionary
    with open(tissue_dict_file, 'r') as f:
        tissue_dict = json.load(f)
    
    # Load gene sequences
    gene_sequences = {}
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('>'):
                gene_id = line.strip()[1:]
                if i + 1 < len(lines):
                    gene_sequences[gene_id] = lines[i + 1].strip()
    
    return fasta_headers, ig_data, seqs, motif_info, tissue_dict, gene_sequences

def plot_global_gradients(gene_idx, gene_id, grads_saliency, tissue_dict, 
                         motif_positions=None, save_dir="output"):
    """Plot global gradient saliency for all tissues"""
    
    n_tissue = grads_saliency.shape[3]
    fig, axes = plt.subplots(n_tissue, 1, figsize=(12, n_tissue * 0.8))
    if n_tissue == 1:
        axes = [axes]
    
    tissue_colors = sns.color_palette("tab20", n_tissue)
    
    # Calculate global min/max for consistent scaling
    gene_grads = grads_saliency[gene_idx, :, :, :]
    min_val = np.min(np.sum(gene_grads, axis=1))
    max_val = np.max(np.sum(gene_grads, axis=1))
    max_abs_val = max(np.abs(min_val), np.abs(max_val))
    min_val = -max_abs_val * 1.1
    max_val = max_abs_val * 1.1
    
    for ti in range(n_tissue):
        score = gene_grads[:, :, ti]
        score_sum = np.sum(score, axis=-1)
        
        # Plot gradient line
        axes[ti].plot(np.arange(len(score_sum)), score_sum, 
                     linewidth=1, color=tissue_colors[ti], 
                     label=tissue_dict[str(ti)])
        
        # Mark motif positions if provided
        if motif_positions:
            for motif_pos in motif_positions:
                axes[ti].axvline(x=motif_pos, color='red', linestyle='--', 
                               linewidth=1, alpha=0.6)
        
        axes[ti].set_xlim(0, len(score_sum))
        axes[ti].set_ylim(min_val, max_val)
        axes[ti].legend(fontsize=8, loc='upper right')
        axes[ti].set_yticks([])
        axes[ti].set_xticks([])
        
        # Remove spines for cleaner look
        for spine in axes[ti].spines.values():
            spine.set_visible(False)
    
    # Add title and labels
    axes[0].set_title(f"Global Gradient Saliency: {gene_id}", fontsize=14, pad=20)
    axes[-1].set_xlabel("Position (32,768 bp)", fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{gene_id}_global_gradients.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/{gene_id}_global_gradients.pdf", dpi=300, bbox_inches='tight')
    
    return fig

def plot_motif_comprehensive_view(gene_idx, gene_id, grads_saliency, gene_sequence, 
                                 motif_info, tissue_dict, logo_width=192, save_dir="output", 
                                 output_format="pdf", fig_width=20, y_min=-0.15, y_max=0.15,
                                 mode='max-grad', symmetric_y=True, percentile_clip=None):
    """Plot comprehensive view: global gradients + combined sequence logo for all motifs organized by tissue
    
    Args:
        mode: 'true-seq' or 'max-grad' - determines how to display sequence logos
        symmetric_y: If True, use symmetric y-axis around 0
        percentile_clip: If set, clip y values at percentile (e.g., 95 clips outliers)
    """
    
    gene_motifs = motif_info.get(gene_id, [])
    if not gene_motifs:
        print(f"No motifs found for gene {gene_id}")
        return None
    
    n_motifs = len(gene_motifs)
    sequence_length = grads_saliency.shape[1]  # 32768
    
    # Create figure organized by tissue: Tissue 0 section + Tissue 1 section  
    # Each tissue section: 1 global + 1 combined logo (not per motif)
    rows_per_tissue = 2  # 1 global + 1 combined logo
    total_rows = 2 * rows_per_tissue  # 2 tissues
    
    # Height ratios: tissue 0 (global, logo) + tissue 1 (global, logo)
    tissue_ratios = [1.5, 1.2]  # global + logo
    height_ratios = tissue_ratios + tissue_ratios  # duplicate for both tissues
    
    fig, axes = plt.subplots(total_rows, 1, 
                           figsize=(fig_width, 6),
                           gridspec_kw={'height_ratios': height_ratios})
    
    # Calculate combined local region for all motifs
    motif_positions = [(motif['absolute_start'], motif['absolute_end']) for motif in gene_motifs]
    all_starts = [pos[0] for pos in motif_positions]
    all_ends = [pos[1] for pos in motif_positions]
    
    # Find the center of all motifs
    leftmost_start = min(all_starts)
    rightmost_end = max(all_ends)
    combined_center = (leftmost_start + rightmost_end) // 2
    
    # Calculate initial flanking region
    plot_start = max(0, combined_center - logo_width // 2)
    plot_end = min(sequence_length, combined_center + logo_width // 2)
    
    # Extend region if any motif is outside the flanking area
    for start, end in motif_positions:
        if start < plot_start:
            plot_start = max(0, start - 20)  # Add 20bp buffer
        if end > plot_end:
            plot_end = min(sequence_length, end + 20)  # Add 20bp buffer
    
    print(f"Combined motif region: {plot_start}-{plot_end} (width: {plot_end - plot_start}bp)")
    print(f"Contains {n_motifs} motifs:")
    for i, motif in enumerate(gene_motifs):
        rel_start = motif['absolute_start'] - plot_start
        rel_end = motif['absolute_end'] - plot_start
        print(f"  Motif {i+1}: {motif['motif_id']} at relative position {rel_start}:{rel_end}")
    
    # Extract reference sequence for the combined region (for true-seq mode)
    local_ref_seq = None
    if mode == 'true-seq' and gene_sequence:
        local_ref_seq = gene_sequence[plot_start:plot_end]
    
    # Collect motif regions for global marking
    motif_regions = [(plot_start, plot_end, "Combined")]
    
    # Plot by tissue organization
    gene_grads = grads_saliency[gene_idx, :, :, :]  # Shape: (32768, 4, 23)
    
    for tissue_idx in range(2):  # Process tissue 0, then tissue 1
        tissue_start_row = tissue_idx * rows_per_tissue
        
        # Plot global gradient for this tissue
        ax_global = axes[tissue_start_row]
        score = gene_grads[:, :, tissue_idx]
        score_sum = np.sum(score, axis=-1)
        
        # Plot gradient line in black
        ax_global.plot(np.arange(len(score_sum)), score_sum, 
                      linewidth=1, color='black')
        
        # Mark combined motif region with light blue shadow
        ax_global.axvspan(plot_start, plot_end, alpha=0.3, color='lightblue')
        
        # Add red center line for global gradients
        center_pos = sequence_length // 2
        ax_global.axvline(x=center_pos, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # Styling for global
        ax_global.set_xlim(0, sequence_length)
        ax_global.set_xticks([])
        ax_global.set_yticks([])
        
        # Add gene ID and tissue name annotation in top-left corner (pure text, no line break)
        tissue_name = tissue_dict.get(str(tissue_idx), f"Tissue_{tissue_idx}")
        ax_global.text(0.015, 0.95, f"{gene_id}  {tissue_name}", 
                      transform=ax_global.transAxes, fontsize=10, 
                      verticalalignment='top', horizontalalignment='left')
        
        # Remove spines and add 0 line
        ax_global.spines['top'].set_visible(False)
        ax_global.spines['right'].set_visible(False)
        ax_global.spines['bottom'].set_visible(False)
        ax_global.spines['left'].set_linewidth(0.5)
        ax_global.axhline(y=0, color='black', linewidth=0.5)
        
        # Plot combined sequence logo for this tissue
        logo_row = tissue_start_row + 1
        ax_logo = axes[logo_row]
        
        # Extract local gradients for combined logo
        local_grads = grads_saliency[gene_idx, plot_start:plot_end, :, :]
        logo_grads = local_grads[:, :, tissue_idx]
        
        # Create sequence logo with mode support
        plot_seq_scores_on_axis(
            logo_grads,
            ax=ax_logo,
            plot_y_ticks=False,
            y_min=y_min,
            y_max=y_max,
            center_marker=False,
            title=None,
            remove_spines=True,
            mode=mode,
            ref_seq=local_ref_seq,
            symmetric_y=symmetric_y,
            percentile_clip=percentile_clip
        )
        
        # Mark each motif region with light red shadow and add position annotation
        for motif in gene_motifs:
            motif_start_rel = motif['absolute_start'] - plot_start
            motif_end_rel = motif['absolute_end'] - plot_start
            ax_logo.axvspan(motif_start_rel, motif_end_rel, alpha=0.3, color='lightcoral')
            
            # Add position annotation
            motif_center_rel = (motif_start_rel + motif_end_rel) // 2
            # Convert to relative position from sequence center (16384)
            center = sequence_length // 2
            rel_pos_start = motif['absolute_start'] - center
            rel_pos_end = motif['absolute_end'] - center
            
            # Add text annotation above the motif region (use motif name, no line break)
            ax_logo.annotate(f'{motif["motif_name"]} ({rel_pos_start}:{rel_pos_end})', 
                           xy=(motif_center_rel, ax_logo.get_ylim()[1]), 
                           xytext=(motif_center_rel, ax_logo.get_ylim()[1] + 0.1 * (ax_logo.get_ylim()[1] - ax_logo.get_ylim()[0])),
                           ha='center', va='bottom', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.3),
                           arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.7))
        
        # Remove all spines and ticks
        for spine in ax_logo.spines.values():
            spine.set_visible(False)
        ax_logo.set_xticks([])
        ax_logo.set_yticks([])
        ax_logo.axhline(y=0., color='black', linestyle='-', linewidth=0.5)
    
    # Remove title as requested
    plt.tight_layout()
    
    # Save in requested format
    os.makedirs(save_dir, exist_ok=True)
    if output_format.lower() == 'svg':
        plt.savefig(f"{save_dir}/{gene_id}_comprehensive_analysis.svg", 
                   dpi=300, bbox_inches='tight', format='svg')
    elif output_format.lower() == 'png':
        plt.savefig(f"{save_dir}/{gene_id}_comprehensive_analysis.png", 
                   dpi=300, bbox_inches='tight', format='png')
    else:
        plt.savefig(f"{save_dir}/{gene_id}_comprehensive_analysis.pdf", 
                   dpi=300, bbox_inches='tight')
    
    return fig

def process_single_gene(args):
    """Process a single gene for multiprocessing"""
    gene_idx, gene_id, gene_sequence, gene_motifs, grads_data, tissue_dict, save_dir, output_format, fig_width, y_min, y_max, mode, symmetric_y, percentile_clip = args
    
    # Force matplotlib to use Agg backend in worker process
    import matplotlib
    matplotlib.use('Agg', force=True)
    
    start_time = time.time()
    
    try:
        if gene_motifs and gene_sequence:
            comprehensive_fig = plot_motif_comprehensive_view(
                gene_idx, gene_id, grads_data, gene_sequence,
                {gene_id: gene_motifs}, tissue_dict, save_dir=save_dir, output_format=output_format,
                fig_width=fig_width, y_min=y_min, y_max=y_max,
                mode=mode, symmetric_y=symmetric_y, percentile_clip=percentile_clip
            )
            if comprehensive_fig:
                plt.close(comprehensive_fig)
            
            elapsed = time.time() - start_time
            return f"✓ {gene_id}: {len(gene_motifs)} motifs ({elapsed:.1f}s)"
        elif not gene_sequence:
            return f"⚠ {gene_id}: No sequence found"
        else:
            return f"⚠ {gene_id}: No motifs found"
    except Exception as e:
        elapsed = time.time() - start_time
        return f"✗ {gene_id}: Error - {str(e)} ({elapsed:.1f}s)"

def visualize_all_genes(max_genes=None, save_dir="output", output_format="pdf",
                       h5_file='all_genes.h5', motif_json='motif_info.json', 
                       fasta_file='all_genes.fa', tissue_dict_file='../23tissues_dict.json',
                       n_processes=48, fig_width=20, y_min=-0.15, y_max=0.15,
                       mode='max-grad', symmetric_y=True, percentile_clip=None):
    """Main function to visualize all genes using multiprocessing
    
    Args:
        mode: 'true-seq' or 'max-grad' - determines how to display sequence logos
        symmetric_y: If True, use symmetric y-axis around 0
        percentile_clip: If set, clip y values at percentile (e.g., 95 clips outliers)
    """
    
    print("Starting motif visualization...")
    print(f"Display mode: {mode}")
    print(f"Symmetric Y-axis: {symmetric_y}")
    if percentile_clip:
        print(f"Percentile clipping: {percentile_clip}%")
    
    # Load data
    fasta_headers, grads_saliency, seqs, motif_info, tissue_dict, gene_sequences = load_data(
        h5_file=h5_file, motif_json=motif_json, 
        fasta_file=fasta_file, tissue_dict_file=tissue_dict_file)
    
    # Decode gene IDs
    gene_ids = [header.decode('utf-8') for header in fasta_headers]
    
    print(f"Found {len(gene_ids)} genes")
    print(f"Gradient data shape: {grads_saliency.shape}")
    print(f"Available tissues: {len(tissue_dict)}")
    print(f"Using {n_processes} processes")
    
    # Limit number of genes if specified
    if max_genes:
        gene_ids = gene_ids[:max_genes]
        print(f"Processing first {max_genes} genes")
    
    # Prepare arguments for multiprocessing - only pass necessary data for each gene
    args_list = []
    genes_with_motifs = 0
    
    for gene_idx, gene_id in enumerate(gene_ids):
        gene_motifs = motif_info.get(gene_id, [])
        gene_sequence = gene_sequences.get(gene_id, "")
        
        if gene_motifs and gene_sequence:
            # Only extract gradient data for this specific gene to reduce memory usage
            gene_grads = grads_saliency[gene_idx:gene_idx+1, :, :, :]  # Keep 4D shape but only 1 gene
            args_list.append((
                0, gene_id, gene_sequence, gene_motifs, 
                gene_grads, tissue_dict, save_dir, output_format, fig_width, y_min, y_max,
                mode, symmetric_y, percentile_clip
            ))
            genes_with_motifs += 1
        elif gene_motifs:
            args_list.append((
                gene_idx, gene_id, "", gene_motifs, 
                None, tissue_dict, save_dir, output_format, fig_width, y_min, y_max,
                mode, symmetric_y, percentile_clip
            ))
        else:
            args_list.append((
                gene_idx, gene_id, gene_sequence, [], 
                None, tissue_dict, save_dir, output_format, fig_width, y_min, y_max,
                mode, symmetric_y, percentile_clip
            ))
    
    print(f"Found {genes_with_motifs} genes with motifs to process")
    
    # Process genes in parallel
    print(f"\nProcessing {len(gene_ids)} genes using {n_processes} processes...")
    start_time = time.time()
    
    if n_processes == 1:
        # Single process mode for debugging
        results = []
        for i, args in enumerate(args_list):
            print(f"Processing gene {i+1}/{len(args_list)}: {args[1]}")
            result = process_single_gene(args)
            results.append(result)
            print(f"  {result}")
    else:
        # Multi-process mode
        print("Starting parallel processing...")
        with Pool(processes=n_processes) as pool:
            results = pool.map(process_single_gene, args_list)
    
    total_time = time.time() - start_time
    
    # Print summary
    print(f"\nVisualization completed! Results saved in '{save_dir}' directory")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Average time per gene: {total_time/len(gene_ids):.1f}s")
    print(f"Genes with motifs processed: {genes_with_motifs}")
    print("\nProcessing Summary:")
    for result in results:
        print(f"  {result}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize motif gradients for genes')
    parser.add_argument('--max_genes', type=int, default=None, 
                       help='Maximum number of genes to process (default: all)')
    parser.add_argument('--output_dir', type=str, default='motif_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--logo_width', type=int, default=192,
                       help='Width of local motif view in bp')
    parser.add_argument('--output_format', type=str, default='pdf', choices=['pdf', 'svg', 'png'],
                       help='Output format: pdf, svg, or png (default: pdf)')
    parser.add_argument('--h5_file', type=str, default='all_genes.h5',
                       help='Path to HDF5 file with gradient data (default: all_genes.h5)')
    parser.add_argument('--motif_json', type=str, default='motif_info.json',
                       help='Path to motif information JSON file (default: motif_info.json)')
    parser.add_argument('--fasta_file', type=str, default='all_genes.fa',
                       help='Path to FASTA file with gene sequences (default: all_genes.fa)')
    parser.add_argument('--tissue_dict', type=str, default='23tissues_modified_dict.json',
                       help='Path to tissue dictionary JSON file (default: %default)')
    parser.add_argument('-p', '--processes', type=int, default=48,
                       help='Number of processes for parallel processing (default: %default)')
    parser.add_argument('--fig_width', type=float, default=18,
                       help='Figure width in inches for global gradients (default: %default)')
    parser.add_argument('--y_min', type=float, default=-0.025,
                       help='Minimum Y-axis limit for sequence logos (default: %default)')
    parser.add_argument('--y_max', type=float, default=0.025,
                       help='Maximum Y-axis limit for sequence logos (default: %default)')
    parser.add_argument('--mode', type=str, default='true-seq', choices=['true-seq', 'max-grad'],
                       help='Display mode: "true-seq" shows actual sequence with attribution, '
                            '"max-grad" shows nucleotide with max |attribution| (default: max-grad)')
    parser.add_argument('--symmetric_y', type=lambda x: x.lower() in ['true', '1', 'yes'], 
                       default=True,
                       help='Use symmetric y-axis around 0 (default: True)')
    parser.add_argument('--percentile_clip', type=float, default=None,
                       help='Clip y-axis values at specified percentile (e.g., 95) to reduce outlier effects')
    
    args = parser.parse_args()
    
    # Run visualization
    visualize_all_genes(max_genes=args.max_genes, save_dir=args.output_dir, 
                       output_format=args.output_format, h5_file=args.h5_file,
                       motif_json=args.motif_json, fasta_file=args.fasta_file,
                       tissue_dict_file=args.tissue_dict, n_processes=args.processes,
                       fig_width=args.fig_width, y_min=args.y_min, y_max=args.y_max,
                       mode=args.mode, symmetric_y=args.symmetric_y, 
                       percentile_clip=args.percentile_clip)
