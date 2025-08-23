#!/usr/bin/env python3
"""
Motif Metadata Summary and Integration Script

This script integrates motif discovery results from multiple tissues, combining:
1. MoDISco H5 files with seqlet data
2. TOMTOM TSV files with motif annotations
3. Tissue-specific saliency score calculations
4. Statistical analysis and significance testing

Usage:
    python motif_metadata_summary.py \
        --modisco_dir modisco_results \
        --tomtom_dir tomtom_results \
        --tissue_indices "1,19,20,21,23" \
        --output integrated_motifs.h5

Author: Generated for motif analysis pipeline
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import ranksums, spearmanr
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MotifIntegrator:
    """
    Integrates motif discovery results from multiple tissues into a unified H5 file.
    """
    
    def __init__(self, tissue_indices: List[int], grads_dir: str = None):
        self.tissue_indices = tissue_indices
        self.n_tissues = len(tissue_indices)
        self.motif_data = defaultdict(dict)
        self.jaspar_to_motifs = defaultdict(list)
        self.jaspar_pwms = {}  # Store JASPAR PWMs
        self.multiple_jaspar_motifs = {} # record the multiple JASPAR motifs if
        self.grads_dir = grads_dir  # Directory containing 5tissues_grads files
        self.grads_data = {}  # Cache for loaded grads data 

    def load_grads_data(self, tissue_id: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load seqs and grads data from 5tissues_grads/tissue_{tissue_id}.h5
        
        Args:
            tissue_id: Tissue identifier
            
        Returns:
            Dictionary containing seqs and grads arrays, or None if file not found
        """
        if tissue_id in self.grads_data:
            return self.grads_data[tissue_id]
        
        if not self.grads_dir:
            logger.warning("No grads directory specified")
            return None
            
        grads_file = os.path.join(self.grads_dir, f"tissue_{tissue_id}.h5")
        
        if not os.path.exists(grads_file):
            logger.warning(f"Grads file not found: {grads_file}")
            return None
            
        try:
            with h5py.File(grads_file, 'r') as f:
                grads_data = {
                    'seqs': np.array(f['seqs']),      # shape: (n_examples, seq_len, 4)
                    'grads': np.array(f['grads'])     # shape: (n_examples, seq_len, 4, n_tissues)
                }
                
                logger.info(f"Loaded grads data for tissue {tissue_id}: seqs {grads_data['seqs'].shape}, grads {grads_data['grads'].shape}")
                
                # Cache the data
                self.grads_data[tissue_id] = grads_data
                return grads_data
                
        except Exception as e:
            logger.error(f"Error loading grads file {grads_file}: {e}")
            return None

    def load_jaspar_meme_file(self, meme_file_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load JASPAR PWMs from MEME format file.
        
        Args:
            meme_file_path: Path to JASPAR MEME file
            
        Returns:
            Dictionary mapping JASPAR ID to motif information
        """
        jaspar_data = {}
        
        if not os.path.exists(meme_file_path):
            logger.warning(f"JASPAR MEME file not found: {meme_file_path}")
            return jaspar_data
        
        try:
            with open(meme_file_path, 'r') as f:
                lines = f.readlines()
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Look for MOTIF lines
                if line.startswith('MOTIF '):
                    parts = line.split()
                    if len(parts) >= 3:
                        jaspar_id = parts[1]  # e.g., MA0021.1
                        motif_name = parts[2] if len(parts) > 2 else jaspar_id  # e.g., Dof3
                        
                        # Look for the letter-probability matrix
                        i += 1
                        while i < len(lines) and not lines[i].strip().startswith('letter-probability matrix:'):
                            i += 1
                        
                        if i < len(lines):
                            # Parse matrix header
                            matrix_line = lines[i].strip()
                            # Extract width from header like "letter-probability matrix: alength= 4 w= 6 nsites= 21 E= 0"
                            width = None
                            parts = matrix_line.split()
                            for j, part in enumerate(parts):
                                if part == 'w=' and j + 1 < len(parts):
                                    width = int(parts[j + 1])
                                    break
                                elif part.startswith('w=') and len(part) > 2:
                                    # Handle case like "w=6" (no space)
                                    width = int(part[2:])
                                    break
                            
                            if width is not None:
                                # Read PWM matrix
                                pwm_matrix = []
                                i += 1
                                for _ in range(width):
                                    if i < len(lines):
                                        matrix_row = lines[i].strip()
                                        if matrix_row and not matrix_row.startswith('URL') and not matrix_row.startswith('MOTIF'):
                                            try:
                                                # Parse probability values (A C G T)
                                                probs = [float(x) for x in matrix_row.split()]
                                                if len(probs) == 4:
                                                    pwm_matrix.append(probs)
                                                else:
                                                    break
                                            except ValueError:
                                                break
                                        else:
                                            break
                                        i += 1
                                    else:
                                        break
                                
                                if pwm_matrix:
                                    jaspar_data[jaspar_id] = {
                                        'motif_name': motif_name,
                                        'jaspar_id': jaspar_id,
                                        'pwm': np.array(pwm_matrix),
                                        'width': len(pwm_matrix)
                                    }
                
                i += 1
            
            logger.info(f"Loaded {len(jaspar_data)} JASPAR motifs from {meme_file_path}")
            self.jaspar_pwms = jaspar_data
            return jaspar_data
            
        except Exception as e:
            logger.error(f"Error loading JASPAR MEME file {meme_file_path}: {e}")
            return {}
        
    def load_tomtom_results(self, tomtom_dir: str) -> Dict[int, pd.DataFrame]:
        """
        Load TOMTOM results for all tissues.
        
        Args:
            tomtom_dir: Directory containing TOMTOM TSV files
            
        Returns:
            Dictionary mapping tissue_id to DataFrame with TOMTOM results
        """
        tomtom_data = {}
        
        for tissue_id in self.tissue_indices:
            tsv_file = os.path.join(tomtom_dir, f"tissue_{tissue_id}_bestHit_motif.tsv")
            
            if not os.path.exists(tsv_file):
                logger.warning(f"TOMTOM file not found: {tsv_file}")
                continue
                
            try:
                df = pd.read_csv(tsv_file, sep='\t')
                logger.info(f"Loaded TOMTOM results for tissue {tissue_id}: {len(df)} entries")
                tomtom_data[tissue_id] = df
                
                # Build JASPAR to motif mapping
                for _, row in df.iterrows():
                    if len(row['Motif_JASPAR'].split(",")) > 1:
                        logger.warning(f"Multiple JASPAR motifs found for {row['Query_ID']}: {row['Motif_JASPAR']}. \n"
                                       "We have not checked this case yet. Please check manually. Default to use the first one.")
                        motif_jaspar = row['Motif_JASPAR'].split(",")[0]
                        # record the multiple JASPAR motifs, and print at the end
                        self.multiple_jaspar_motifs[row['Query_ID']] = row['Motif_JASPAR']
                    else:
                        motif_jaspar = row['Motif_JASPAR']
                        
                    query_id = row['Query_ID']
                    
                    # Try to match with JASPAR database using Target_ID
                    target_id = row.get('Target_ID', '')
                    matched_jaspar_id = None
                    
                    # First try exact match with Target_ID
                    if target_id in self.jaspar_pwms:
                        matched_jaspar_id = target_id
                    # Then try to find by motif name
                    elif motif_jaspar:
                        for jid, jdata in self.jaspar_pwms.items():
                            if jdata['motif_name'].upper() == motif_jaspar.upper():
                                matched_jaspar_id = jid
                                break
                    
                    # Use the matched JASPAR ID if found, otherwise use the original jaspar motif name
                    final_jaspar_id = matched_jaspar_id if matched_jaspar_id else motif_jaspar
                    
                    # Store both the original and matched JASPAR ID in the row
                    row_with_match = row.copy()
                    row_with_match['matched_jaspar_id'] = final_jaspar_id
                    
                    self.jaspar_to_motifs[final_jaspar_id].append((tissue_id, query_id, row_with_match))
                    
            except Exception as e:
                logger.error(f"Error loading TOMTOM file {tsv_file}: {e}")
                
        return tomtom_data
    
    def load_modisco_data(self, modisco_dir: str, tissue_id: int) -> Optional[h5py.File]:
        """
        Load MoDISco H5 file for a specific tissue.
        
        Args:
            modisco_dir: Directory containing MoDISco H5 files
            tissue_id: Tissue identifier
            
        Returns:
            Opened H5 file handle or None if file not found
        """
        h5_file = os.path.join(modisco_dir, f"modisco_tissue_{tissue_id}.h5")
        
        if not os.path.exists(h5_file):
            logger.warning(f"MoDISco file not found: {h5_file}")
            return None
            
        try:
            return h5py.File(h5_file, 'r')
        except Exception as e:
            logger.error(f"Error opening MoDISco file {h5_file}: {e}")
            return None
    
    def extract_seqlet_data(self, modisco_h5: h5py.File, query_id: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract seqlet data for a specific query ID from MoDISco H5 file.
        
        Args:
            modisco_h5: Opened MoDISco H5 file
            query_id: Query ID (e.g., "neg_patterns.pattern_0")
            
        Returns:
            Dictionary containing seqlet data arrays
        """
        try:
            # Parse query ID to get pattern type and number
            if query_id.startswith('neg_patterns.pattern_'):
                pattern_type = 'neg_patterns'
                pattern_num = query_id.split('pattern_')[1]
            elif query_id.startswith('pos_patterns.pattern_'):
                pattern_type = 'pos_patterns'
                pattern_num = query_id.split('pattern_')[1]
            else:
                logger.warning(f"Unknown query ID format: {query_id}")
                return None
            
            # Navigate to the pattern in H5 structure
            pattern_path = f"{pattern_type}/pattern_{pattern_num}"
            
            if pattern_path not in modisco_h5:
                logger.warning(f"Pattern not found in H5: {pattern_path}")
                return None
            
            pattern_group = modisco_h5[pattern_path]
            
            # Extract main pattern data
            seqlet_data = {
                'contrib_scores': np.array(pattern_group['contrib_scores']),
                'hypothetical_contribs': np.array(pattern_group['hypothetical_contribs']),
                'sequence': np.array(pattern_group['sequence'])
            }
            
            # Extract seqlets data if available
            if 'seqlets' in pattern_group:
                seqlets_group = pattern_group['seqlets']
                seqlet_data.update({
                    'seqlets_contrib_scores': np.array(seqlets_group['contrib_scores']),
                    'seqlets_hypothetical_contribs': np.array(seqlets_group['hypothetical_contribs']),
                    'seqlets_sequence': np.array(seqlets_group['sequence']),
                    'seqlets_start': np.array(seqlets_group['start']),
                    'seqlets_end': np.array(seqlets_group['end']),
                    'seqlets_example_idx': np.array(seqlets_group['example_idx']),
                    'seqlets_is_revcomp': np.array(seqlets_group['is_revcomp']),
                    'n_seqlets': np.array(seqlets_group['n_seqlets'])[0] if 'n_seqlets' in seqlets_group else len(seqlets_group['start'])
                })
            
            return seqlet_data
            
        except Exception as e:
            logger.error(f"Error extracting seqlet data for {query_id}: {e}")
            return None
    
    def merge_seqlet_data(self, all_seqlet_data: List[Tuple[str, Dict[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
        """
        Merge seqlet data from multiple queries for the same tissue.
        
        Args:
            all_seqlet_data: List of (query_id, seqlet_data) tuples
            
        Returns:
            Merged seqlet data dictionary
        """
        if not all_seqlet_data:
            return {}
        
        # Initialize merged data with first query
        _, first_data = all_seqlet_data[0]
        merged_data = {}
        
        # Keys that should be concatenated along the first axis (seqlets)
        concat_keys = [
            'seqlets_contrib_scores', 'seqlets_hypothetical_contribs', 'seqlets_sequence',
            'seqlets_start', 'seqlets_end', 'seqlets_example_idx', 'seqlets_is_revcomp'
        ]
        
        # Keys that should be averaged or taken from the first query
        single_keys = ['contrib_scores', 'hypothetical_contribs', 'sequence']
        
        # Process concatenation keys
        for key in concat_keys:
            arrays_to_concat = []
            for query_id, seqlet_data in all_seqlet_data:
                if key in seqlet_data:
                    arrays_to_concat.append(seqlet_data[key])
            
            if arrays_to_concat:
                merged_data[key] = np.concatenate(arrays_to_concat, axis=0)
        
        # Process single value keys (use first available)
        for key in single_keys:
            for query_id, seqlet_data in all_seqlet_data:
                if key in seqlet_data:
                    merged_data[key] = seqlet_data[key]
                    break
        
        # Calculate total number of seqlets
        if 'seqlets_contrib_scores' in merged_data:
            merged_data['n_seqlets'] = merged_data['seqlets_contrib_scores'].shape[0]
        else:
            merged_data['n_seqlets'] = sum(
                seqlet_data.get('n_seqlets', 0) for _, seqlet_data in all_seqlet_data
            )
        
        return merged_data
    
    def calculate_tissue_saliency_scores(self, seqlet_data: Dict[str, np.ndarray], 
                                       current_tissue_id: int) -> Dict[str, Any]:
        """
        Calculate saliency scores for seqlets from current tissue using grads data.
        
        Gradient saliency is calculated as input × gradient from 5tissues_grads files.
        
        Supports two grads layouts:
        - 4D: (n_examples, seq_len, 4, n_tissues) → selects current tissue along last axis
        - 3D: (n_examples, seq_len, 4) → already for the current tissue
        
        Args:
            seqlet_data: Dictionary containing seqlet data (with example indices)
            current_tissue_id: ID of the current tissue being processed
            
        Returns:
            Dictionary containing saliency_scores array and gradient_saliency_seqlets list
        """
        # Load grads data for this tissue
        grads_data = self.load_grads_data(current_tissue_id)
        if grads_data is None:
            logger.warning(f"No grads data available for tissue {current_tissue_id}")
            return {
                'saliency_scores': np.array([]),
                'gradient_saliency_seqlets': []
            }
        
        # Check if we have seqlet indices
        if 'seqlets_example_idx' not in seqlet_data or 'seqlets_start' not in seqlet_data or 'seqlets_end' not in seqlet_data:
            logger.warning("No seqlet position data available for saliency calculation")
            return {
                'saliency_scores': np.array([]),
                'gradient_saliency_seqlets': []
            }
        
        seqlets_example_idx = seqlet_data['seqlets_example_idx']
        seqlets_start = seqlet_data['seqlets_start']
        seqlets_end = seqlet_data['seqlets_end']
        
        seqs = grads_data['seqs']      # shape: (n_examples, seq_len, 4)
        grads = grads_data['grads']    # shape: (n_examples, seq_len, 4[, n_tissues])
        
        # Determine gradient tensor shape and select current tissue if needed
        if grads.ndim == 4:
            # Map current_tissue_id to its position in self.tissue_indices
            tissue_idx = None
            for i, tid in enumerate(self.tissue_indices):
                if tid == current_tissue_id:
                    tissue_idx = i
                    break
            if tissue_idx is None:
                logger.warning(f"Tissue {current_tissue_id} not found in tissue indices")
                return {
                    'saliency_scores': np.array([]),
                    'gradient_saliency_seqlets': []
                }
            tissue_grads = grads[:, :, :, tissue_idx]  # (n_examples, seq_len, 4)
        elif grads.ndim == 3:
            tissue_grads = grads  # Already for this tissue
        else:
            logger.warning(f"Unexpected grads ndim={grads.ndim} for tissue {current_tissue_id}")
            return {
                'saliency_scores': np.array([]),
                'gradient_saliency_seqlets': []
            }
        
        saliency_scores = []
        gradient_saliency_seqlets = []  # Store full gradient saliency for each seqlet
        
        # Calculate saliency for each seqlet
        for i in range(len(seqlets_example_idx)):
            example_idx = seqlets_example_idx[i]
            start_pos = seqlets_start[i]
            end_pos = seqlets_end[i]
            
            # Extract sequence and gradient for this seqlet
            seqlet_seq = seqs[example_idx, start_pos:end_pos, :]      # shape: (seqlet_len, 4)
            seqlet_grad = tissue_grads[example_idx, start_pos:end_pos, :]  # shape: (seqlet_len, 4)
            
            # Calculate gradient saliency as input × gradient
            gradient_saliency = seqlet_seq * seqlet_grad  # shape: (seqlet_len, 4)
            
            # Store the full gradient saliency for this seqlet
            gradient_saliency_seqlets.append(gradient_saliency)
            
            # Calculate mean absolute saliency across sequence and nucleotides for summary score
            saliency_score = np.mean(np.abs(gradient_saliency))
            saliency_scores.append(saliency_score)
        
        logger.info(f"Calculated {len(saliency_scores)} saliency scores for tissue {current_tissue_id} using grads data")
        
        # Return both summary scores and full gradient saliency arrays
        return {
            'saliency_scores': np.array(saliency_scores),
            'gradient_saliency_seqlets': gradient_saliency_seqlets
        }
    
    def calculate_saliency_statistics(self, saliency_scores: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for saliency scores from a single tissue.
        
        Args:
            saliency_scores: Array of shape (n_seqlets,) with saliency scores
            
        Returns:
            Dictionary containing various statistics
        """
        if saliency_scores.size == 0:
            return {}
        
        stats = {
            'mean': float(np.mean(saliency_scores)),
            'std': float(np.std(saliency_scores)),
            'median': float(np.median(saliency_scores)),
            'q05': float(np.quantile(saliency_scores, 0.05)),
            'q50': float(np.quantile(saliency_scores, 0.50)),  # Same as median
            'q80': float(np.quantile(saliency_scores, 0.80)),
            'q95': float(np.quantile(saliency_scores, 0.95)),
            'min': float(np.min(saliency_scores)),
            'max': float(np.max(saliency_scores)),
            'n_seqlets': len(saliency_scores)
        }
        
        return stats
    
    def perform_cross_tissue_wilcoxon_test(self, motif_tissue_scores: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Perform Wilcoxon rank-sum test between tissues with largest and second largest 95th percentile.
        This follows the method described in the paper.
        
        Args:
            motif_tissue_scores: Dictionary mapping tissue_id to saliency scores array
            
        Returns:
            Dictionary containing test statistics
        """
        if len(motif_tissue_scores) < 2:
            return {}
        
        # Calculate 95th percentile for each tissue
        tissue_q95 = {}
        for tissue_id, scores in motif_tissue_scores.items():
            if len(scores) > 0:
                tissue_q95[tissue_id] = np.quantile(scores, 0.95)
        
        if len(tissue_q95) < 2:
            return {}
        
        # Find tissues with largest and second largest 95th percentile
        sorted_tissues = sorted(tissue_q95.items(), key=lambda x: x[1], reverse=True)
        largest_tissue_id, largest_q95 = sorted_tissues[0]
        second_largest_tissue_id, second_largest_q95 = sorted_tissues[1]
        
        # Get the saliency scores for these two tissues
        largest_scores = motif_tissue_scores[largest_tissue_id]
        second_largest_scores = motif_tissue_scores[second_largest_tissue_id]
        
        # Perform two-sided Wilcoxon rank-sum test
        try:
            s_val, p_val = ranksums(
                largest_scores, 
                second_largest_scores, 
                alternative='two-sided'
            )
            
            return {
                'p_value': float(p_val),
                's_value': float(s_val),
                'largest_tissue_id': int(largest_tissue_id),
                'second_largest_tissue_id': int(second_largest_tissue_id),
                'largest_q95': float(largest_q95),
                'second_largest_q95': float(second_largest_q95),
                'largest_n_seqlets': len(largest_scores),
                'second_largest_n_seqlets': len(second_largest_scores)
            }
        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return {}
    
    def calculate_cross_tissue_analysis(self, all_saliency_scores: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Calculate cross-tissue analysis metrics.
        
        Args:
            all_saliency_scores: Dictionary mapping motif names to tissue_id -> saliency_scores dict
            
        Returns:
            Dictionary containing cross-tissue analysis results
        """
        if not all_saliency_scores:
            return {}
        
        # Collect statistics for each tissue across all motifs
        tissue_stats = defaultdict(list)  # tissue_id -> list of motif statistics
        
        for motif_name, tissue_scores in all_saliency_scores.items():
            for tissue_id, scores in tissue_scores.items():
                if len(scores) > 0:
                    tissue_stats[tissue_id].append({
                        'mean': np.mean(scores),
                        'median': np.median(scores),
                        'q95': np.quantile(scores, 0.95),
                        'motif_name': motif_name
                    })
        
        if not tissue_stats:
            return {}
        
        # Calculate summary statistics per tissue
        tissue_summary = {}
        for tissue_id, motif_stats in tissue_stats.items():
            if motif_stats:
                tissue_summary[tissue_id] = {
                    'n_motifs': len(motif_stats),
                    'mean_q95': np.mean([s['q95'] for s in motif_stats]),
                    'median_q95': np.median([s['q95'] for s in motif_stats]),
                    'std_q95': np.std([s['q95'] for s in motif_stats])
                }
        
        # Create tissue comparison matrix
        tissue_ids = sorted(tissue_summary.keys())
        n_tissues = len(tissue_ids)
        
        if n_tissues < 2:
            return {'tissue_summary': tissue_summary}
        
        comparison_matrix = np.zeros((n_tissues, n_tissues))
        
        for i, tissue_i in enumerate(tissue_ids):
            for j, tissue_j in enumerate(tissue_ids):
                if i != j:
                    # Calculate difference in mean 95th percentiles
                    diff = abs(tissue_summary[tissue_i]['mean_q95'] - tissue_summary[tissue_j]['mean_q95'])
                    comparison_matrix[i, j] = diff
        
        return {
            'tissue_summary': tissue_summary,
            'tissue_ids': tissue_ids,
            'q95_difference_matrix': comparison_matrix
        }
    
    def integrate_motifs(self, modisco_dir: str, tomtom_dir: str, output_file: str, jaspar_meme_file: Optional[str] = None):
        """
        Main integration function that combines all motif data.
        
        Args:
            modisco_dir: Directory containing MoDISco H5 files
            tomtom_dir: Directory containing TOMTOM TSV files
            output_file: Output H5 file path
            jaspar_meme_file: Optional path to JASPAR MEME file
        """
        logger.info("Starting motif integration...")
        
        # Load JASPAR PWMs if provided
        if jaspar_meme_file:
            logger.info(f"Loading JASPAR PWMs from: {jaspar_meme_file}")
            self.load_jaspar_meme_file(jaspar_meme_file)
        
        # Load TOMTOM results
        tomtom_data = self.load_tomtom_results(tomtom_dir)
        
        if not tomtom_data:
            logger.error("No TOMTOM data loaded. Exiting.")
            return
        
        # Create output H5 file
        with h5py.File(output_file, 'w') as out_h5:
            
            # Create main groups
            motifs_group = out_h5.create_group('motifs')
            global_metadata_group = out_h5.create_group('global_metadata')
            tissue_summary_group = out_h5.create_group('tissue_summary')
            
            # Store global metadata
            global_metadata_group.create_dataset('tissues_analyzed', data=np.array(self.tissue_indices))
            global_metadata_group.create_dataset('processing_date', data=np.string_(pd.Timestamp.now().isoformat()))
            
            # Track all saliency scores for cross-tissue analysis
            all_saliency_scores = {}
            
            # Process each JASPAR motif
            for jaspar_id, motif_entries in self.jaspar_to_motifs.items():
                logger.info(f"Processing JASPAR motif, JAPAR_ID: {jaspar_id}, JASPAR_MOTIF: {motif_entries[0][2]['Motif_JASPAR']}")

                # Create motif group
                motif_group = motifs_group.create_group(jaspar_id)
                metadata_group = motif_group.create_group('metadata')
                tissues_group = motif_group.create_group('tissues')
                
                # Store basic metadata
                first_entry = motif_entries[0][2]  # Get first TOMTOM row
                metadata_group.create_dataset('jaspar_id', data=np.string_(jaspar_id))
                metadata_group.create_dataset('consensus_sequence', data=np.string_(first_entry.get('Target_consensus', '')))
                
                # Store JASPAR PWM if available
                if jaspar_id in self.jaspar_pwms:
                    jaspar_info = self.jaspar_pwms[jaspar_id]
                    metadata_group.create_dataset('jaspar_pwm', data=jaspar_info['pwm'])
                    metadata_group.create_dataset('jaspar_motif_name', data=np.string_(jaspar_info['motif_name']))
                    metadata_group.create_dataset('jaspar_width', data=jaspar_info['width'])
                    logger.info(f"  Saved JASPAR PWM for {jaspar_id} with shape: {jaspar_info['pwm'].shape}")
                else:
                    logger.warning(f"  No JASPAR PWM found for {jaspar_id}")
                
                total_seqlets = 0
                tissues_present = []
                motif_tissue_scores = {}  # tissue_id -> saliency_scores array
                
                # Group entries by tissue to handle multiple queries per tissue
                tissue_entries = defaultdict(list)
                for tissue_id, query_id, tomtom_row in motif_entries:
                    tissue_entries[tissue_id].append((query_id, tomtom_row))
                
                # Process each tissue for this motif
                for tissue_id, queries_and_rows in tissue_entries.items():
                    logger.info(f"  Processing tissue {tissue_id} with {len(queries_and_rows)} queries")
                    
                    # Load MoDISco data
                    modisco_h5 = self.load_modisco_data(modisco_dir, tissue_id)
                    if modisco_h5 is None:
                        continue
                    
                    try:
                        # Create tissue group
                        tissue_group = tissues_group.create_group(f'tissue_{tissue_id}')
                        
                        # Collect all query IDs for this tissue
                        query_ids = [query_id for query_id, _ in queries_and_rows]
                        tissue_group.create_dataset('query_ids', data=np.string_(query_ids))
                        
                        # Collect seqlet data from all queries for this tissue
                        all_seqlet_data = []
                        all_tomtom_rows = []
                        
                        for query_id, tomtom_row in queries_and_rows:
                            logger.info(f"    Processing query {query_id}")
                            
                            # Extract seqlet data
                            seqlet_data = self.extract_seqlet_data(modisco_h5, query_id)
                            if seqlet_data is not None:
                                all_seqlet_data.append((query_id, seqlet_data))
                                all_tomtom_rows.append(tomtom_row)
                        
                        if not all_seqlet_data:
                            continue
                        
                        # Merge seqlet data from all queries
                        merged_seqlet_data = self.merge_seqlet_data(all_seqlet_data)
                        
                        # Create seqlets subgroup
                        seqlets_group = tissue_group.create_group('seqlets')
                        
                        # Store merged seqlet data
                        for key, data in merged_seqlet_data.items():
                            if isinstance(data, np.ndarray) and data.size > 0:
                                seqlets_group.create_dataset(key, data=data)
                        
                        # Calculate tissue saliency scores
                        saliency_result = self.calculate_tissue_saliency_scores(merged_seqlet_data, tissue_id)
                        saliency_scores = saliency_result['saliency_scores']
                        gradient_saliency_seqlets = saliency_result['gradient_saliency_seqlets']
                        
                        # Store summary saliency scores
                        if saliency_scores.size > 0:
                            seqlets_group.create_dataset('tissue_saliency_scores', data=saliency_scores)
                            motif_tissue_scores[tissue_id] = saliency_scores
                            
                            # Store full gradient saliency for each seqlet
                            if gradient_saliency_seqlets:
                                # Convert list of arrays to a single array with padding if needed
                                max_len = max(gs.shape[0] for gs in gradient_saliency_seqlets)
                                padded_gradient_saliency = np.zeros((len(gradient_saliency_seqlets), max_len, 4))
                                
                                for i, gs in enumerate(gradient_saliency_seqlets):
                                    padded_gradient_saliency[i, :gs.shape[0], :] = gs
                                
                                seqlets_group.create_dataset('gradient_saliency_seqlets', data=padded_gradient_saliency)
                                seqlets_group.create_dataset('seqlet_lengths', data=np.array([gs.shape[0] for gs in gradient_saliency_seqlets]))
                                
                                logger.info(f"  Stored gradient saliency for {len(gradient_saliency_seqlets)} seqlets with max length {max_len}")
                        else:
                            logger.warning(f"  No saliency scores calculated for tissue {tissue_id}")
                        
                        # Calculate statistics
                        stats_group = tissue_group.create_group('statistics')
                        
                        # Basic counts
                        n_seqlets = merged_seqlet_data.get('n_seqlets', 0)
                        stats_group.create_dataset('n_seqlets', data=n_seqlets)
                        total_seqlets += n_seqlets
                        
                        # Saliency statistics
                        if tissue_id in motif_tissue_scores and motif_tissue_scores[tissue_id].size > 0:
                            saliency_stats = self.calculate_saliency_statistics(motif_tissue_scores[tissue_id])
                            if saliency_stats:
                                saliency_stats_group = stats_group.create_group('saliency_stats')
                                for stat_name, stat_data in saliency_stats.items():
                                    saliency_stats_group.create_dataset(stat_name, data=stat_data)
                            
                            # Note: Wilcoxon tests will be performed at motif level after all tissues are processed
                        
                        # TOMTOM statistics (use best p-value from all queries)
                        tomtom_stats_group = stats_group.create_group('tomtom_stats')
                        best_tomtom = min(all_tomtom_rows, key=lambda x: x.get('p-value', 1.0))
                        tomtom_stats_group.create_dataset('p_value', data=best_tomtom.get('p-value', np.nan))
                        tomtom_stats_group.create_dataset('e_value', data=best_tomtom.get('E-value', np.nan))
                        tomtom_stats_group.create_dataset('q_value', data=best_tomtom.get('q-value', np.nan))
                        tomtom_stats_group.create_dataset('overlap', data=best_tomtom.get('Overlap', 0))
                        
                        # Pattern mapping for all queries
                        pattern_mapping_group = tissue_group.create_group('pattern_mapping')
                        
                        for query_id, seqlet_data in all_seqlet_data:
                            query_group = pattern_mapping_group.create_group(query_id)
                            
                            # Store original pattern data
                            for key in ['contrib_scores', 'hypothetical_contribs', 'sequence']:
                                if key in seqlet_data:
                                    query_group.create_dataset(f'original_{key}', data=seqlet_data[key])
                        
                        tissues_present.append(tissue_id)
                        
                    finally:
                        modisco_h5.close()
                
                # Update motif metadata
                metadata_group.create_dataset('total_seqlets_count', data=total_seqlets)
                metadata_group.create_dataset('tissues_present', data=np.array(tissues_present))
                
                # Perform cross-tissue Wilcoxon test for this motif (following paper methodology)
                if len(motif_tissue_scores) >= 2:
                    wilcoxon_results = self.perform_cross_tissue_wilcoxon_test(motif_tissue_scores)
                    if wilcoxon_results:
                        cross_tissue_group = motif_group.create_group('cross_tissue_analysis')
                        wilcoxon_group = cross_tissue_group.create_group('wilcoxon_test')
                        for test_name, test_data in wilcoxon_results.items():
                            wilcoxon_group.create_dataset(test_name, data=test_data)
                        
                        logger.info(f"  Wilcoxon test: p-value={wilcoxon_results.get('p_value', 'N/A'):.2e}, "
                                  f"tissues {wilcoxon_results.get('largest_tissue_id')} vs {wilcoxon_results.get('second_largest_tissue_id')}")
                
                # Store tissue scores for global analysis
                if motif_tissue_scores:
                    all_saliency_scores[jaspar_id] = motif_tissue_scores
            
            # Global cross-tissue analysis
            logger.info("Calculating global cross-tissue analysis...")
            global_cross_tissue = self.calculate_cross_tissue_analysis(all_saliency_scores)
            
            if global_cross_tissue:
                global_cross_group = out_h5.create_group('global_cross_tissue_analysis')
                
                # Save tissue IDs
                if 'tissue_ids' in global_cross_tissue:
                    global_cross_group.create_dataset('tissue_ids', data=np.array(global_cross_tissue['tissue_ids']))
                
                # Save comparison matrix
                if 'q95_difference_matrix' in global_cross_tissue:
                    global_cross_group.create_dataset('q95_difference_matrix', data=global_cross_tissue['q95_difference_matrix'])
                
                # Save tissue summary as separate datasets
                if 'tissue_summary' in global_cross_tissue:
                    tissue_summary = global_cross_tissue['tissue_summary']
                    summary_group = global_cross_group.create_group('tissue_summary')
                    
                    for tissue_id, stats in tissue_summary.items():
                        tissue_stats_group = summary_group.create_group(f'tissue_{tissue_id}')
                        for stat_name, stat_value in stats.items():
                            tissue_stats_group.create_dataset(stat_name, data=stat_value)
            
            # Update global metadata
            global_metadata_group.create_dataset('total_motifs_found', data=len(self.jaspar_to_motifs))
            
            # Tissue summary
            for tissue_id in self.tissue_indices:
                tissue_group = tissue_summary_group.create_group(f'tissue_{tissue_id}')
                
                # Count patterns and motifs for this tissue
                total_patterns = 0
                annotated_patterns = 0
                unique_motifs = set()
                
                if tissue_id in tomtom_data:
                    total_patterns = len(tomtom_data[tissue_id])
                    annotated_patterns = len(tomtom_data[tissue_id])
                    unique_motifs = set(tomtom_data[tissue_id]['Motif_JASPAR'].values)
                
                tissue_group.create_dataset('total_patterns', data=total_patterns)
                tissue_group.create_dataset('annotated_patterns', data=annotated_patterns)
                tissue_group.create_dataset('unique_motifs', data=len(unique_motifs))
        if self.multiple_jaspar_motifs:
            pairs = list(zip(self.multiple_jaspar_motifs.keys(), self.multiple_jaspar_motifs.values()))
            logger.warning(f"Found the following queries have multiple JASPAR motifs. Please check manually. Default to use the first one.\n"
                           f"{pairs}")
        logger.info(f"Motif integration completed. Output saved to: {output_file}")


def main():
    """Main function to run the motif integration script."""
    parser = argparse.ArgumentParser(description='Integrate motif discovery results from multiple tissues')
    
    parser.add_argument('--modisco_dir', required=True, 
                       help='Directory containing MoDISco H5 files')
    parser.add_argument('--tomtom_dir', required=True,
                       help='Directory containing TOMTOM TSV files')
    parser.add_argument('--tissue_indices', required=True,
                       help='Comma-separated list of tissue indices (e.g., "1,19,20,21,23")')
    parser.add_argument('--output', required=True,
                       help='Output H5 file path')
    parser.add_argument('--grads_dir', required=True,
                       help='Directory containing 5tissues_grads files (e.g., 5tissues_grads/)')
    parser.add_argument('--jaspar_meme', 
                       help='Path to JASPAR MEME file (e.g., JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse tissue indices
    try:
        tissue_indices = [int(x.strip()) for x in args.tissue_indices.split(',')]
    except ValueError:
        logger.error("Invalid tissue indices format. Use comma-separated integers.")
        sys.exit(1)
    
    # Validate input directories
    if not os.path.exists(args.modisco_dir):
        logger.error(f"MoDISco directory not found: {args.modisco_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.tomtom_dir):
        logger.error(f"TOMTOM directory not found: {args.tomtom_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.grads_dir):
        logger.error(f"Grads directory not found: {args.grads_dir}")
        sys.exit(1)

    # Run integration
    integrator = MotifIntegrator(tissue_indices, args.grads_dir)
    integrator.integrate_motifs(args.modisco_dir, args.tomtom_dir, args.output, args.jaspar_meme)


if __name__ == '__main__':
    main()
