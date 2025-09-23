#!/usr/bin/env python3
"""
Motif Visualization Pipeline

This script creates comprehensive visualizations for motifs including:
1. Sequence logos based on information content
2. Saliency score distributions across tissues
3. Pie charts showing tissue-specific seqlet proportions
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
import argparse
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from functools import partial
import json
import warnings
warnings.filterwarnings('ignore')

# 硬编码的色板 - 美观的颜色组合
TISSUE_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2',
    '#A9CCE3', '#FAD7A0', '#ABEBC6', '#F9E79F', '#D5A6BD',
    '#A2D9CE', '#F7DC6F', '#BB8FCE'
]

# Helper function to draw DNA letters
def dna_letter_at(letter, x, y, yscale=1, ax=None, color=None, alpha=1.0):
    """Draw a DNA letter at a given position."""
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    globscale = 1.35
    LETTERS = {
        "T": TextPath((-0.305, 0), "T", size=1, prop=fp),
        "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
        "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
        "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
    }
    COLOR_SCHEME = {
        'G': 'orange',
        'A': 'green',
        'C': 'blue',
        'T': 'red',
    }

    text = LETTERS[letter]
    chosen_color = COLOR_SCHEME[letter] if color is None else color

    t = transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        transforms.Affine2D().translate(x, y) + ax.transData
    p = PathPatch(text, lw=0, fc=chosen_color, alpha=alpha, transform=t)
    if ax is not None:
        ax.add_artist(p)
    return p

def calculate_information_content(ppm: np.ndarray) -> np.ndarray:
    """
    Calculate information content for each position in PPM.
    
    Args:
        ppm: Position probability matrix [L, 4]
    
    Returns:
        Information content array [L]
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    ppm_safe = ppm + eps
    ppm_safe = ppm_safe / np.sum(ppm_safe, axis=1, keepdims=True)
    
    # Calculate entropy: H_i = -sum(p_i,b * log2(p_i,b))
    entropy = -np.sum(ppm_safe * np.log2(ppm_safe + eps), axis=1)
    
    # Information content: I_i = log2(4) - H_i = 2 - H_i
    info_content = 2.0 - entropy
    
    return info_content

def plot_sequence_logo(ppm: np.ndarray, ax=None, title: str = "", y_max: float = 2.0):
    """
    Plot sequence logo based on PPM information content.
    
    Args:
        ppm: Position probability matrix [L, 4]
        ax: Matplotlib axis
        title: Plot title (not used anymore)
        y_max: Maximum y-axis value (shared between both logos for consistent scale)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(max(4, ppm.shape[0] * 0.3), 2))
    
    # Calculate information content
    info_content = calculate_information_content(ppm)
    
    # Plot sequence logo with reduced proportions
    height_base = 0.0
    logo_height = 1.0  # Restore to normal height, we'll control size via layout ratios
    
    # Loop over sequence positions
    for j in range(ppm.shape[0]):
        # Sort nucleotides by probability
        sort_index = np.argsort(ppm[j, :])
        
        # Loop over nucleotide identities (from bottom to top)
        for ii in range(4):
            i = sort_index[ii]
            
            # Calculate height proportional to information content
            nt_prob = ppm[j, i] * info_content[j]
            
            # Decode letter from nucleotide identity
            nt = ['A', 'C', 'G', 'T'][i]
            
            # Draw letter at position
            if ii == 0:
                dna_letter_at(nt, j + 0.5, height_base, nt_prob * logo_height, ax)
            else:
                prev_prob = np.sum(ppm[j, sort_index[:ii]]) * logo_height * info_content[j]
                dna_letter_at(nt, j + 0.5, height_base + prev_prob, nt_prob * logo_height, ax)
    
    # Set axis properties
    ax.set_xlim(0, ppm.shape[0])
    ax.set_ylim(0, y_max)
    ax.set_xticks([])
    ax.set_yticks([2.0])  # Only show tick at 2.0
    ax.set_ylabel('')  # Remove "Bits" label
    ax.set_title('')  # Remove title
    
    # Set Arial font for y-axis ticks
    ax.tick_params(axis='y', labelsize=12, labelcolor='black')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')  # Use Arial font
    
    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Keep bottom border for x-axis and left border for y-axis
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

def plot_saliency_distribution(saliency_data: Dict[int, np.ndarray], 
                             tissue_names: List[str],
                             ax=None,
                             n_total: int = 0,
                             polarity_info: Dict = None,
                             min_limit: float = -0.015,
                             max_limit: float = 0.015,
                             show_y_axis: bool = False):
    """
    Plot saliency score distribution across tissues.
    
    Args:
        saliency_data: Dictionary mapping tissue index to saliency scores
        tissue_names: List of tissue names
        ax: Matplotlib axis
        n_total: Total number of seqlets
        polarity_info: Dictionary with polarity and p-value information
        min_limit: Minimum y-axis limit
        max_limit: Maximum y-axis limit
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    # Create mapping from tissue indices to positions
    tissue_indices = sorted(saliency_data.keys())
    tissue_to_pos = {tissue_idx: pos for pos, tissue_idx in enumerate(tissue_indices)}
    
    # Plot individual points with transparency
    for tissue_idx, scores in saliency_data.items():
        if len(scores) > 0:
            pos = tissue_to_pos[tissue_idx]
            # Create jittered x positions
            x_pos = pos + np.random.normal(0, 0.1, len(scores))
            color_idx = pos % len(TISSUE_COLORS)  # Use position instead of tissue_idx for color order
            ax.scatter(x_pos, scores, alpha=0.2, s=10, color='black', edgecolors='none')
            
            # Calculate and plot percentiles
            p05 = np.percentile(scores, 5)
            p50 = np.percentile(scores, 50)
            p95 = np.percentile(scores, 95)
            
            # Plot percentile lines
            ax.plot([pos - 0.3, pos + 0.3], [p05, p05], 
                   color=TISSUE_COLORS[color_idx], linewidth=2, alpha=0.8)
            ax.plot([pos - 0.3, pos + 0.3], [p50, p50], 
                   color=TISSUE_COLORS[color_idx], linewidth=3, alpha=0.9)
            ax.plot([pos - 0.3, pos + 0.3], [p95, p95], 
                   color=TISSUE_COLORS[color_idx], linewidth=2, alpha=0.8)
            
            # Add shadows between percentile lines
            ax.fill_between([pos - 0.3, pos + 0.3], [p05, p05], [p50, p50], 
                          color=TISSUE_COLORS[color_idx], alpha=0.2)
            ax.fill_between([pos - 0.3, pos + 0.3], [p50, p50], [p95, p95], 
                          color=TISSUE_COLORS[color_idx], alpha=0.2)
            
            # Add percentile markers
            ax.scatter([pos - 0.3], [p05], color=TISSUE_COLORS[color_idx], 
                      s=50, marker='D', edgecolors='black', linewidth=1, alpha=0.8)
            ax.scatter([pos], [p50], color=TISSUE_COLORS[color_idx], 
                      s=60, marker='o', edgecolors='black', linewidth=1, alpha=0.9)
            ax.scatter([pos + 0.3], [p95], color=TISSUE_COLORS[color_idx], 
                      s=50, marker='D', edgecolors='black', linewidth=1, alpha=0.8)
    
    # Set axis properties
    ax.set_xlim(-0.5, len(tissue_indices) - 0.5)
    ax.set_ylim(min_limit, max_limit)
    ax.set_xticks(range(len(tissue_indices)))
    # Use actual tissue names if available, otherwise use indices
    tick_labels = []
    for tissue_idx in tissue_indices:
        if tissue_idx < len(tissue_names):
            tick_labels.append(tissue_names[tissue_idx])
        else:
            tick_labels.append(f"Tissue_{tissue_idx}")
    
    # Set x-axis labels with corresponding tissue colors
    for i, (tissue_idx, label) in enumerate(zip(tissue_indices, tick_labels)):
        color_idx = i % len(TISSUE_COLORS)  # Use position i instead of tissue_idx for color order
        ax.text(i, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, 
               label, ha='center', va='top', color=TISSUE_COLORS[color_idx], 
               fontsize=14, fontfamily='Arial')  # Use Arial font
    
    # Add x-axis ticks
    ax.set_xticks(range(len(tissue_indices)))
    ax.set_xticklabels([])  # No labels on ticks, we have colored labels below
    
    # Control y-axis visibility
    if not show_y_axis:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)
    else:
        ax.set_ylabel('Saliency Score', fontfamily='Arial')  # Optionally add a label when shown
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    text_x = 0.45
    text_font_size=12
    text_y = 0.98
    # Add annotations for sample size and p-values
    ax.text(text_x, text_y, f'$n$ = {n_total:,}', 
            transform=ax.transAxes, ha='left', va='top', 
            fontsize=text_font_size, fontfamily='Arial')
    # Display p-values based on polarity
    if polarity_info and polarity_info.get('p95_significant', False):
        p_act = polarity_info.get('p95_pvalue', 1.0)
        top_tissue_p95 = polarity_info.get('top_tissue_p95')
        color = 'black' # Default color
        if top_tissue_p95 is not None and top_tissue_p95 in tissue_to_pos:
            pos = tissue_to_pos[top_tissue_p95]
            color = TISSUE_COLORS[pos % len(TISSUE_COLORS)]
            ax.text(text_x, text_y -0.05 , f'$P_{{act}}$ ≤ {p_act:.2e}',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=text_font_size, fontfamily='Arial', color=color)
            if polarity_info.get('p05_significant', False):
                p_rep = polarity_info.get('p05_pvalue', 1.0)
                top_tissue_p05 = polarity_info.get('top_tissue_p05')
                color = 'black' # Default color
                if top_tissue_p05 is not None and top_tissue_p05 in tissue_to_pos:
                    pos = tissue_to_pos[top_tissue_p05]
                    color = TISSUE_COLORS[pos % len(TISSUE_COLORS)]
                    ax.text(text_x, text_y - 0.1, f'$P_{{rep}}$ ≤ {p_rep:.2e}',
                            transform=ax.transAxes, ha='left', va='top',
                            fontsize=text_font_size, fontfamily='Arial', color=color)
        

    elif polarity_info and polarity_info.get('p05_significant', False):
        p_rep = polarity_info.get('p05_pvalue', 1.0)
        top_tissue_p05 = polarity_info.get('top_tissue_p05')
        color = 'black' # Default color
        if top_tissue_p05 is not None and top_tissue_p05 in tissue_to_pos:
            pos = tissue_to_pos[top_tissue_p05]
            color = TISSUE_COLORS[pos % len(TISSUE_COLORS)]
    
        ax.text(text_x, text_y - 0.05, f'$P_{{rep}}$ ≤ {p_rep:.2e}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=text_font_size, fontfamily='Arial', color=color)

def plot_seqlet_histogram(tissue_data: Dict[int, Dict], 
                          tissue_names: List[str],
                          ax=None):
    """
    Plot normalized histogram showing seqlet counts per tissue.
    
    Args:
        tissue_data: Dictionary mapping tissue index to tissue data with n_seqlets
        tissue_names: List of tissue names
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    
    # Get all tissue indices that have data
    tissue_indices = sorted(tissue_data.keys())
    
    # Collect seqlet counts
    counts = []
    for tissue_idx in tissue_indices:
        tissue_info = tissue_data.get(tissue_idx, {})
        n_seqlets = tissue_info.get('n_seqlets', 0)
        counts.append(n_seqlets)
    
    # Normalize counts to [0, 1] range
    max_count = max(counts) if counts else 1
    normalized_counts = [count / max_count for count in counts] if max_count > 0 else counts
    
    # Create bar positions matching the distribution plot
    tissue_to_pos = {tissue_idx: pos for pos, tissue_idx in enumerate(tissue_indices)}
    positions = [tissue_to_pos[tissue_idx] for tissue_idx in tissue_indices]
    
    # Plot bars
    bars = []
    for i, (tissue_idx, norm_count, count) in enumerate(zip(tissue_indices, normalized_counts, counts)):
        color_idx = i % len(TISSUE_COLORS)  # Use position i instead of tissue_idx for color order
        bar = ax.bar(positions[i], norm_count, color=TISSUE_COLORS[color_idx], 
                    alpha=0.7, edgecolor='black', linewidth=0.5)
        bars.append(bar)
        
        # Add count labels on top of bars
        if norm_count > 0:
            ax.text(positions[i], norm_count + 0.01, str(count), 
                   ha='center', va='bottom', fontsize=8, fontweight='bold', 
                   fontfamily='Arial')  # Use Arial font
    
    # Set axis properties
    ax.set_xlim(-0.5, len(tissue_indices) - 0.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([])  # Remove x-axis labels (shared with distribution plot below)
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_ylabel('')  # Remove y-axis label
    ax.set_title('')  # Remove title
    ax.set_ylim(0, 1.3)

    # Remove borders except bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def load_motif_data(h5_file: str, motif_name: str) -> Dict:
    """
    Load data for a specific motif from H5 file.
    
    Args:
        h5_file: Path to H5 file
        motif_name: Name of the motif
    
    Returns:
        Dictionary containing motif data
    """
    data = {}
    
    with h5py.File(h5_file, 'r') as f:
        motif_group = f[f'motifs/{motif_name}']
        
        # Load PPMs
        if 'best_query_ppm' in motif_group:
            data['best_query_ppm'] = motif_group['best_query_ppm'][:]
        if 'target_ppm' in motif_group:
            data['target_ppm'] = motif_group['target_ppm'][:]
        
        # Load attributes
        data['best_evalue'] = motif_group.attrs.get('best_evalue', 1.0)
        data['target_id'] = motif_group.attrs.get('target_id', 'Unknown')
        
        # Load tissue data
        data['tissues'] = {}
        tissues_group = motif_group['tissues']
        
        for tissue_name in tissues_group.keys():
            tissue_group = tissues_group[tissue_name]
            tissue_idx = int(tissue_name.split('_')[1])
            
            data['tissues'][tissue_idx] = {
                'n_seqlets': tissue_group.attrs.get('saliency_n_seqlets', 0),
                'total_seqlets': tissue_group.attrs.get('saliency_total_seqlets', 0),
                'p05': tissue_group.attrs.get('saliency_p05', 0.0),
                'p50': tissue_group.attrs.get('saliency_p50', 0.0),
                'p95': tissue_group.attrs.get('saliency_p95', 0.0)
            }
    
    return data

def load_saliency_scores(h5_file: str, motif_name: str) -> Dict[int, np.ndarray]:
    """
    Load individual saliency scores for a motif.
    
    Args:
        h5_file: Path to H5 file
        motif_name: Name of the motif
    
    Returns:
        Dictionary mapping tissue index to saliency scores
    """
    saliency_data = {}
    
    with h5py.File(h5_file, 'r') as f:
        if 'analysis/seqlet_saliency_scores' in f:
            ss_group = f['analysis/seqlet_saliency_scores']
            
            # Load data
            jaspar_names = [name.decode('utf-8') for name in ss_group['Jaspar_Name'][:]]
            tissues = ss_group['Tissue'][:]
            scores = ss_group['Saliency_Score'][:]
            
            # Filter for this motif
            mask = np.array(jaspar_names) == motif_name
            motif_tissues = tissues[mask]
            motif_scores = scores[mask]
            
            # Group by tissue
            for tissue_idx in np.unique(motif_tissues):
                tissue_mask = motif_tissues == tissue_idx
                saliency_data[int(tissue_idx)] = motif_scores[tissue_mask]
    
    return saliency_data



def load_top_hits_info(h5_file: str, motif_name: str) -> Dict:
    """
    Load polarity and p-value information for a specific motif from the 'top_hits' table.
    
    Args:
        h5_file: Path to H5 file
        motif_name: Name of the motif
    
    Returns:
        Dictionary with polarity and significance information.
    """
    result = {
        'polarity': 'none',
        'p95_significant': False,
        'p05_significant': False,
        'p95_pvalue': 1.0,
        'p05_pvalue': 1.0,
        'top_tissue_p95': None,
        'top_tissue_p05': None
    }
    
    with h5py.File(h5_file, 'r') as f:
        if 'analysis/top_hits' in f:
            th_group = f['analysis/top_hits']
            
            # Decode Jaspar names
            jaspar_names = [name.decode('utf-8') for name in th_group['Jaspar_Name'][:]]
            
            try:
                # Find the index for the current motif
                idx = jaspar_names.index(motif_name)
                
                # Load data for the specific motif using the found index
                result['polarity'] = th_group['Polarity'][idx].decode('utf-8')
                result['p95_significant'] = bool(th_group['P95_Significant'][idx])
                result['p05_significant'] = bool(th_group['P05_Significant'][idx])
                
                p95_pval = th_group['P95_P_value'][idx]
                p05_pval = th_group['P05_P_value'][idx]
                
                result['p95_pvalue'] = float(p95_pval) if not np.isnan(p95_pval) else 1.0
                result['p05_pvalue'] = float(p05_pval) if not np.isnan(p05_pval) else 1.0

                # Load top tissue indices
                if 'Top_Tissue_P95' in th_group.keys():
                    top_t_p95 = th_group['Top_Tissue_P95'][idx]
                    if not np.isnan(top_t_p95):
                        result['top_tissue_p95'] = int(top_t_p95)

                if 'Top_Tissue_P05' in th_group.keys():
                    top_t_p05 = th_group['Top_Tissue_P05'][idx]
                    if not np.isnan(top_t_p05):
                        result['top_tissue_p05'] = int(top_t_p05)
            
            except ValueError:
                # Motif not found in top_hits, return default values
                pass
    
    return result

def create_motif_visualization(h5_file: str, 
                             motif_name: str, 
                             tissue_names: List[str],
                             output_dir: str,
                             min_limit: float = -0.015,
                             max_limit: float = 0.015,
                             show_y_axis: bool = False,
                             image_format: str = "png"):
    """
    Create complete visualization for a single motif.
    
    Args:
        h5_file: Path to H5 file
        motif_name: Name of the motif
        tissue_names: List of tissue names
        output_dir: Output directory for saving plots
        min_limit: Minimum y-axis limit for distribution plot
        max_limit: Maximum y-axis limit for distribution plot
    """
    try:
        # Load motif data
        motif_data = load_motif_data(h5_file, motif_name)
        saliency_data = load_saliency_scores(h5_file, motif_name)
        polarity_info = load_top_hits_info(h5_file, motif_name)
        
        # Get total seqlets from the first available tissue's metadata, as it's the same for all.
        total_seqlets = 0
        if motif_data.get('tissues'):
            first_tissue_key = next(iter(motif_data['tissues']), None)
            if first_tissue_key is not None:
                total_seqlets = motif_data['tissues'][first_tissue_key].get('total_seqlets', 0)
        
        # Hardcode layout ratios: [info, logo1, logo2, histogram, distribution]
        layout_ratios = [0.4, 0.6, 0.6, 0.5, 4]  # Increased logo2 proportion from 0.7 to 0.8 to add space between logos
        
        # Determine PPM lengths for width calculation
        query_ppm_length = motif_data.get('best_query_ppm', np.zeros((1, 4))).shape[0]
        target_ppm_length = motif_data.get('target_ppm', np.zeros((1, 4))).shape[0]
        max_ppm_length = max(query_ppm_length, target_ppm_length)
        
        # Calculate figure width based on PPM length - keep distribution plot width fixed
        # Logo plots can be compressed, but distribution plot maintains consistent width
        base_width = 3.0  # Fixed base width for consistent distribution plot width
        logo_width = max_ppm_length * 0.08  # Logo width based on PPM length
        fig_width = max(base_width, logo_width)  # Use the larger of the two to ensure logo fits
        
        # Cap the maximum width to prevent overly wide figures
        fig_width = min(fig_width, 4.0)
        
        # Create figure with calculated width and height
        fig = plt.figure(figsize=(fig_width, 9))
        
        # Create grid with 5 rows: info, logo1, logo2, histogram, distribution
        gs = fig.add_gridspec(5, 1, height_ratios=layout_ratios, hspace=0.05)  # Start with no spacing
        
        # Plot sequence logos with same y_max for consistent scale
        y_max = 2.0  # Shared y_max for both logos
        
        # Add motif info at the top - with motif name and E-value prefix, no background
        ax_info = fig.add_subplot(gs[0])
        ax_info.axis('off')
        info_text = f"{motif_name}\nE-value: {motif_data['best_evalue']:.2e}"
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center', 
                    transform=ax_info.transAxes, fontsize=14,
                    color='black', fontfamily='Arial')  # Use Arial font
        
        # First logo
        ax1 = fig.add_subplot(gs[1])
        if 'best_query_ppm' in motif_data:
            plot_sequence_logo(motif_data['best_query_ppm'], ax1, y_max=y_max)
        
        # Second logo
        ax2 = fig.add_subplot(gs[2])
        if 'target_ppm' in motif_data:
            plot_sequence_logo(motif_data['target_ppm'], ax2, y_max=y_max)
        
        # Plot seqlet histogram
        ax_hist = fig.add_subplot(gs[3])
        plot_seqlet_histogram(motif_data['tissues'], tissue_names, ax_hist)
        
        # Plot saliency distribution with shared x-axis
        ax_dist = fig.add_subplot(gs[4], sharex=ax_hist)
        plot_saliency_distribution(saliency_data, tissue_names, 
                                 ax_dist, total_seqlets, polarity_info, min_limit, max_limit,
                                 show_y_axis=show_y_axis)
        

        # Save combined plot
        safe_name = motif_name.replace('/', '_').replace(' ', '_')
        fig.savefig(f"{output_dir}/{safe_name}_combined.{image_format}", dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        
        print(f"Created visualization for {motif_name}")
        
    except Exception as e:
        print(f"Error creating visualization for {motif_name}: {e}")
        import traceback
        traceback.print_exc()

def get_motif_list(h5_file: str) -> List[str]:
    """
    Get list of all motifs in H5 file.
    
    Args:
        h5_file: Path to H5 file
    
    Returns:
        List of motif names
    """
    motif_names = []
    
    with h5py.File(h5_file, 'r') as f:
        if 'motifs' in f:
            motif_names = list(f['motifs'].keys())
    
    return motif_names

def load_tissue_dict(tissue_dict_file: str) -> Dict[str, str]:
    """
    Load tissue dictionary from JSON file.
    
    Args:
        tissue_dict_file: Path to tissue dictionary JSON file
    
    Returns:
        Dictionary mapping tissue indices to names
    """
    try:
        with open(tissue_dict_file, 'r') as f:
            tissue_dict = json.load(f)
        return tissue_dict
    except Exception as e:
        print(f"Warning: Could not load tissue dictionary from {tissue_dict_file}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Create motif visualizations")
    parser.add_argument("--h5_file", required=True, help="Path to H5 file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--tissue_dict", required=True, help="Path to tissue dictionary JSON file")
    parser.add_argument("--n_processes", type=int, default=1, 
                       help="Number of processes for parallel processing")
    parser.add_argument("--motif_filter", default="", 
                       help="Filter motifs by name (optional)")
    parser.add_argument("--min_limit", type=float, default=-0.015,
                       help="Minimum y-axis limit for distribution plot (default: -0.015)")
    parser.add_argument("--max_limit", type=float, default=0.015,
                       help="Maximum y-axis limit for distribution plot (default: 0.015)")
    parser.add_argument("--tissue_colors", default="", 
                       help="Comma-separated list of hex colors for tissues (e.g., #FF0000,#00FF00,#0000FF)")
    parser.add_argument("--show_y_axis", action="store_true", 
                       help="Show y-axis labels and ticks on the distribution plot (default: hidden)")
    parser.add_argument("--image_format", default="png", 
                       help="Output image format (e.g., png, svg, pdf; default: png)")

    
    args = parser.parse_args()
    
    # Handle custom tissue colors
    global TISSUE_COLORS
    if args.tissue_colors:
        custom_colors = args.tissue_colors.split(',')
        if len(custom_colors) > 0:
            TISSUE_COLORS = custom_colors
            print(f"Using custom tissue colors: {TISSUE_COLORS}")
    
    # Load tissue dictionary
    tissue_dict = load_tissue_dict(args.tissue_dict)
    
    # Create tissue names list from dictionary
    tissue_names = []
    max_tissue_idx = max([int(k) for k in tissue_dict.keys()]) if tissue_dict else 0
    for i in range(max_tissue_idx + 1):
        tissue_names.append(tissue_dict.get(str(i), f"Tissue_{i}"))
    
    print(f"Loaded {len(tissue_names)} tissue names from dictionary")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get motif list
    motif_names = get_motif_list(args.h5_file)
    
    # Apply filter if specified
    if args.motif_filter:
        filtered_motifs = []
        for name in motif_names:
            if args.motif_filter in name:
                filtered_motifs.append(name)
        motif_names = filtered_motifs
    
    print(f"Found {len(motif_names)} motifs to visualize")
    
    # Create visualization function with fixed arguments
    viz_func = partial(create_motif_visualization, 
                      args.h5_file, 
                      tissue_names=tissue_names,
                      output_dir=args.output_dir,
                      min_limit=args.min_limit,
                      max_limit=args.max_limit,
                      show_y_axis=args.show_y_axis,
                      image_format=args.image_format)
    
    # Process motifs
    if args.n_processes > 1 and len(motif_names) > 1:
        # Use multiprocessing
        with mp.Pool(args.n_processes) as pool:
            pool.map(viz_func, motif_names)
    else:
        # Sequential processing
        for motif_name in motif_names:
            viz_func(motif_name)
    
    print(f"Visualization complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    # Debug mode
    if len(sys.argv) == 1:
        sys.argv += [
            "--h5_file", "motif_anaylsis_32768_5tissues_atg_fw_grads_results/motif_clusters.h5",
            "--output_dir", "motif_plots_32768_5tissues_atg_fw_grads",
            "--tissue_dict", "23tissues_dict.json",
            "--n_processes", "32",
            "--min_limit", "-0.015",
            "--max_limit", "0.015",
            "--tissue_colors", "#C10606,#C2560F,#196E9B,#8E6A50,#2A386F"
        ]
    
    main()
