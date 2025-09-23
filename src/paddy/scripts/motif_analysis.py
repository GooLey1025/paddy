#!/usr/bin/env python3
"""
Motif Analysis Pipeline

This script implements:
1. Motif matching and deduplication using Tomtom results
2. Merging seqlet coordinates from different tissues
3. Computing tissue-specific saliency scores
4. Analyzing motif tissue specificity
"""

import h5py
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse
import json
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

class MotifAnalysisPipeline:
    def __init__(self, tissue_indices: List[int], grads_dir: str, data_dir: str, output_dir: str, evalue_thresh: float = 0.1, window_size: int = None, up_size: int = None, down_size: int = None):
        self.tissue_indices = [int(ti) for ti in tissue_indices.split(",")]
        self.grads_dir = grads_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.evalue_thresh = evalue_thresh
        self.window_size = window_size
        self.up_size = up_size
        self.down_size = down_size
        
        # Discover tomtom XML files under the provided directory
        self.tomtom_xml_files = []
        for ti in self.tissue_indices:
            tissue_dir = os.path.join(self.data_dir, f"tomtom_tissue_{ti}")
            tomtom_file = os.path.join(tissue_dir, "tomtom.xml")
            if os.path.isdir(tissue_dir) and os.path.exists(tomtom_file):
                self.tomtom_xml_files.append(tomtom_file)
        
        # Data storage
        self.tomtom_matches = None
        self.merged_seqlets = None
        self.tissue_saliency_scores = None
    
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Log window configuration
        if self.window_size is not None:
            if self.up_size is not None or self.down_size is not None:
                up_val = self.up_size if self.up_size is not None else 0
                down_val = self.down_size if self.down_size is not None else 0
                output_window_size = up_val + down_val
                print(f"Windowed preprocessing detected:")
                print(f"  - Center size: {self.window_size} bp")
                print(f"  - Output window: {output_window_size} bp (up={up_val}, down={down_val})")
                print(f"  - Will convert seqlet coordinates from output-window relative to absolute grads coordinates")
            else:
                print(f"Window size specified: {self.window_size} bp - will convert seqlet coordinates from center-window relative to absolute")
        else:
            print("No window size specified - using seqlet coordinates as-is (assumed to be absolute)")

    def convert_seqlet_coordinates(self, start: int, end: int, grads_length: int) -> Tuple[int, int]:
        """
        Convert seqlet coordinates from window-relative to absolute coordinates in grads file.
        
        When using windowed preprocessing with up_size/down_size:
        - seqlets start/end are relative to the output window (up_size + down_size)
        - need to convert first to center window coordinates, then to absolute positions
        
        When using windowed modiscolite without up_size/down_size:
        - seqlets start/end are relative to the center window
        - need to convert to absolute positions in the full grads sequence
        
        Args:
            start: Start position relative to output/center window
            end: End position relative to output/center window
            grads_length: Total length of sequence in grads file
            
        Returns:
            Tuple of (absolute_start, absolute_end) in grads coordinates
        """
        if self.window_size is None:
            # No window size specified, return coordinates as-is
            return start, end
        
        # Calculate center of the full sequence
        grads_center = grads_length // 2
        
        # Two-stage conversion if up_size/down_size are specified
        if self.up_size is not None or self.down_size is not None:
            up_val = self.up_size if self.up_size is not None else 0
            down_val = self.down_size if self.down_size is not None else 0
            
            # Stage 1: Convert from output window coordinates to center window coordinates
            # The output window was extracted from the center of the center_size window
            center_window_half = self.window_size // 2
            output_window_center_in_center = center_window_half  # Output window is centered in center window
            
            # Convert from output window coordinates to center window coordinates
            center_start = output_window_center_in_center - up_val + start
            center_end = output_window_center_in_center - up_val + end
            
            # Stage 2: Convert from center window coordinates to absolute grads coordinates
            center_window_start_in_grads = grads_center - center_window_half
            absolute_start = center_window_start_in_grads + center_start
            absolute_end = center_window_start_in_grads + center_end
            
        else:
            # Original logic: seqlets are relative to center window
            half_window = self.window_size // 2
            absolute_start = grads_center - half_window + start
            absolute_end = grads_center - half_window + end
        
        # Ensure coordinates are within bounds
        absolute_start = max(0, absolute_start)
        absolute_end = min(grads_length, absolute_end)
        
        # Debug logging for first few conversions
        if hasattr(self, '_conversion_count'):
            self._conversion_count += 1
        else:
            self._conversion_count = 1
            
        if self._conversion_count <= 5:  # Log first 5 conversions
            if self.up_size is not None or self.down_size is not None:
                up_val = self.up_size if self.up_size is not None else 0
                down_val = self.down_size if self.down_size is not None else 0
                print(f"  Coordinate conversion {self._conversion_count}: output_window({start},{end}) -> absolute({absolute_start},{absolute_end})")
                print(f"    [grads_length={grads_length}, center_size={self.window_size}, up={up_val}, down={down_val}]")
            else:
                print(f"  Coordinate conversion {self._conversion_count}: center_window({start},{end}) -> absolute({absolute_start},{absolute_end})")
                print(f"    [grads_length={grads_length}, center_size={self.window_size}]")
        
        return absolute_start, absolute_end

    def align_ppms(self, query_ppm: List[List[float]], target_ppm: List[List[float]], offset: int, rc: bool = False) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Align query and target PPMs based on offset and reverse complement logic.
        
        Logic according to user specification:
        - If rc=True, reverse complement the target first
        - Offset is relative to the (potentially reverse complemented) target:
          * Negative offset: query has extra bases at beginning that should be removed
          * Positive offset: query needs padding at beginning to align with target
        - Trim both sequences to shortest length after alignment
        
        Examples:
        - offset=-11, rc=False: query's first 11 bases are extra, start alignment from query[12:]
        - offset=-2, rc=True: reverse complement target, then query's first 2 bases are extra
        - offset=+6, rc=False: query needs 6 bases padding at beginning to align with target
        - offset=+9, rc=True: reverse complement target, then query needs 9 bases padding
        
        Args:
            query_ppm: Query PPM as list of position probability vectors
            target_ppm: Target PPM as list of position probability vectors  
            offset: Offset value (negative = trim query start, positive = pad query start)
            rc: Whether to reverse complement target before alignment
        
        Returns:
            Tuple of (aligned_query_ppm, aligned_target_ppm) with same length
        """
        background = [0.25, 0.25, 0.25, 0.25]
        
        # Step 1: Apply reverse complement to target if rc=True
        working_target = target_ppm.copy()
        if rc:
            # Reverse complement target: reverse sequence and swap A<->T, C<->G
            working_target = working_target[::-1]  # Reverse the sequence
            for i in range(len(working_target)):
                # Swap A<->T, C<->G (A=0, C=1, G=2, T=3)
                working_target[i] = [
                    working_target[i][3],  # T -> A
                    working_target[i][2],  # G -> C  
                    working_target[i][1],  # C -> G
                    working_target[i][0]   # A -> T
                ]
        
        # Step 2: Apply offset-based alignment to query
        working_query = query_ppm.copy()
        
        if offset < 0:
            # Negative offset: query has extra bases at beginning, remove |offset| bases
            trim_amount = abs(offset)
            if trim_amount >= len(working_query):
                # If trimming more than query length, return empty alignment
                working_query = []
            else:
                working_query = working_query[trim_amount:]
        elif offset > 0:
            # Positive offset: query needs padding at beginning to align with target
            padding = [background] * offset
            working_query = padding + working_query
        # offset == 0: no change needed
        
        # Step 3: Trim both sequences to shortest length to ensure proper alignment
        min_length = min(len(working_query), len(working_target))
        
        if min_length == 0:
            # Handle edge case where one sequence becomes empty
            return [], []
        
        aligned_query = working_query[:min_length]
        aligned_target = working_target[:min_length]
        
        return aligned_query, aligned_target

    def parse_tomtom_results(self) -> pd.DataFrame:
        """Parse Tomtom XML results and create a DataFrame of motif matches."""
        print("Parsing Tomtom results...")
        
        all_rows = []
        for xml_file in self.tomtom_xml_files:
            tissue_idx = int(xml_file.split("/")[-2].split("_")[-1])
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Collect query motifs
            query_motifs = []
            for m in root.findall(".//queries/motif"):
                ppm = []
                for pos in m.findall("./pos"):
                    ppm.append([
                        float(pos.attrib["A"]),
                        float(pos.attrib["C"]),
                        float(pos.attrib["G"]),
                        float(pos.attrib["T"]),
                    ])
                query_motifs.append({
                    "id": m.attrib["id"],
                    "db": m.attrib["db"],
                    "length": int(m.attrib["length"]),
                    "ppm": ppm,
                })
            
            # Collect target motifs
            target_motifs = []
            for m in root.findall(".//targets/motif"):
                ppm = []
                for pos in m.findall("./pos"):
                    ppm.append([
                        float(pos.attrib["A"]),
                        float(pos.attrib["C"]),
                        float(pos.attrib["G"]),
                        float(pos.attrib["T"]),
                    ])
                target_motifs.append({
                    "id": m.attrib["id"],
                    "alt": m.attrib["alt"],
                    "db": m.attrib["db"],
                    "length": int(m.attrib["length"]),
                    "ppm": ppm,
                })
            
            # Match and filter
            for q in root.findall(".//matches/query"):
                q_idx = int(q.attrib["idx"])
                if q_idx >= len(query_motifs):
                    continue
                q_id = query_motifs[q_idx]["id"]
                # Parse pattern group/index from Query_ID, e.g., pos_patterns.pattern_3
                pattern_group = None
                pattern_index = None
                try:
                    pg, pidx = q_id.split(".")
                    pattern_group = pg
                    pattern_index = int(pidx.split("_")[-1])
                except Exception:
                    pass
                
                for t in q.findall("./target"):
                    t_idx = int(t.attrib["idx"])
                    if t_idx >= len(target_motifs):
                        continue
                    
                    pv = float(t.attrib["pv"])
                    ev = float(t.attrib["ev"])
                    qv = float(t.attrib["qv"])
                    offset = int(t.attrib["off"])
                    
                    if ev <= self.evalue_thresh:
                        # Align PPMs based on offset
                        query_ppm = query_motifs[q_idx]["ppm"]
                        target_ppm = target_motifs[t_idx]["ppm"]
                        rc = (t.attrib["rc"] == "y")
                        aligned_query_ppm, aligned_target_ppm = self.align_ppms(query_ppm, target_ppm, offset, rc)
                        
                        all_rows.append({
                            "Tissue_Index": tissue_idx,
                            "Query_ID": q_id,
                            "Pattern_Group": pattern_group,
                            "Pattern_Index": pattern_index,
                            "Target_ID": target_motifs[t_idx]["id"],
                            "Jaspar_Name": target_motifs[t_idx]["alt"],
                            "rc": rc,
                            "P_value": pv,
                            "E_value": ev,
                            "Q_value": qv,
                            "Offset": offset,
                            "Query_PPM": aligned_query_ppm,
                            "Target_PPM": aligned_target_ppm,
                            "Aligned_Length": len(aligned_query_ppm)
                        })
        
        df_all = pd.DataFrame(all_rows)
        
        # Keep best match per tissue-query combination
        final_df = df_all.loc[
            df_all.groupby(["Tissue_Index", "Query_ID"])["P_value"].idxmin()
        ].reset_index(drop=True)
        
        self.tomtom_matches = final_df
        print(f"Parsed {len(final_df)} motif matches")
        if self.tomtom_matches is not None:
            self.tomtom_matches.to_csv(f"{self.output_dir}/tomtom_matches.tsv", sep="\t", index=False)
        return final_df
    
    def merge_seqlet_coordinates(self) -> pd.DataFrame:
        """Merge seqlet coordinates from different tissues for the same motifs.
        Returns a DataFrame with all seqlet information and motif metadata.
        """
        print("Merging seqlet coordinates...")
        
        all_seqlets = []
        motif_metadata = {}
        
        # Group by Jaspar motif name
        for jaspar_name, motif_group in self.tomtom_matches.groupby("Jaspar_Name"):
            # Find the best match (lowest P-value) for this motif
            best_match = motif_group.loc[motif_group["P_value"].idxmin()]
            
            # Store motif metadata from the best match
            motif_metadata[jaspar_name] = {
                "target_ppm": np.asarray(best_match["Target_PPM"]) if isinstance(best_match["Target_PPM"], list) else None,
                "target_id": best_match["Target_ID"],
                "best_evalue": float(best_match["E_value"]),
                "best_pvalue": float(best_match["P_value"]),
                "best_query_id": best_match["Query_ID"],
                "best_tissue_idx": int(best_match["Tissue_Index"]),
                "best_query_ppm": np.asarray(best_match["Query_PPM"]) if isinstance(best_match["Query_PPM"], list) else None
            }
            
            # Collect seqlets from each tissue
            for _, row in motif_group.iterrows():
                tissue_idx = row["Tissue_Index"]
                query_id = row["Query_ID"]
                pattern_group = row.get("Pattern_Group", None)
                pattern_index = row.get("Pattern_Index", None)
                
                # Parse query ID to get pattern information
                try:
                    pattern1, pattern2 = query_id.split(".")
                except ValueError:
                    print(f"Warning: Could not parse Query_ID {query_id}")
                    continue
                
                modisco_file = f"{self.data_dir}/modiscolite_tissue_{tissue_idx}.h5"
                
                if not os.path.exists(modisco_file):
                    print(f"Warning: Modisco file not found: {modisco_file}")
                    continue
                
                try:
                    with h5py.File(modisco_file, "r") as f:
                        if pattern1 not in f or pattern2 not in f[pattern1]:
                            print(f"Warning: Pattern {pattern1}/{pattern2} not found in {modisco_file}")
                            continue
                        
                        seqlets_group = f[pattern1][pattern2]["seqlets"]
                        
                        # Extract seqlet information
                        n_seqlets = int(seqlets_group["n_seqlets"][0])
                        example_idx = seqlets_group["example_idx"][:]
                        start = seqlets_group["start"][:]
                        end = seqlets_group["end"][:]
                        # Optional fields
                        is_revcomp = seqlets_group["is_revcomp"][:] if "is_revcomp" in seqlets_group else None
                        contrib_scores = f[pattern1][pattern2]["contrib_scores"][:]
                        hypothetical_contribs = f[pattern1][pattern2]["hypothetical_contribs"][:]
                        sequence = f[pattern1][pattern2]["sequence"][:]
                        
                        # Get grads file length for coordinate conversion if needed
                        grads_length = None
                        if self.window_size is not None:
                            grads_file = f"{self.grads_dir}/tissue_{tissue_idx}.h5"
                            if os.path.exists(grads_file):
                                try:
                                    with h5py.File(grads_file, "r") as grads_f:
                                        grads_length = grads_f["grads_saliency"].shape[1]
                                except Exception:
                                    pass
                        
                        # Create seqlet records for DataFrame
                        for i in range(n_seqlets):
                            # Convert coordinates if window_size is specified
                            abs_start, abs_end = self.convert_seqlet_coordinates(
                                int(start[i]), int(end[i]), grads_length
                            ) if grads_length is not None else (int(start[i]), int(end[i]))
                            
                            seqlet_record = {
                                "tissue_idx": tissue_idx,
                                "example_idx": int(example_idx[i]),
                                "start": int(start[i]),  # Original relative coordinates
                                "end": int(end[i]),      # Original relative coordinates
                                "abs_start": abs_start,  # Absolute coordinates in grads
                                "abs_end": abs_end,      # Absolute coordinates in grads
                                "query_id": query_id,
                                "jaspar_name": jaspar_name,
                                "pattern_group": pattern_group if pattern_group is not None else pattern1,
                                "pattern_index": int(pattern_index) if pattern_index is not None else int(pattern2.split("_")[-1]),
                                "is_revcomp": bool(is_revcomp[i]) if is_revcomp is not None else None,
                                "contrib_score": contrib_scores,
                                "hypothetical_contribs": hypothetical_contribs,
                                "sequence": sequence
                            }
                            all_seqlets.append(seqlet_record)
                        
                except Exception as e:
                    print(f"Error processing {modisco_file} for {query_id}: {e}")
                    continue
        
        # Create DataFrame
        self.merged_seqlets_df = pd.DataFrame(all_seqlets)
        self.motif_metadata = motif_metadata
        
        print(f"Merged seqlets for {len(motif_metadata)} motifs, total {len(all_seqlets)} seqlets")

        return self.merged_seqlets_df

    def save_motif_clusters_h5(self, output_dir: str, output_name: str = "motif_clusters.h5"):
        """Save motif clusters, seqlet metadata, grads_saliency slices, and analysis results to an HDF5 file.
        Now works with DataFrame structure from merge_seqlet_coordinates.

        Layout:
          /motifs/{jaspar_name}/target_ppm [L,4]
          /motifs/{jaspar_name}/target_id (attr)
          /motifs/{jaspar_name}/best_evalue (attr) - E-value of the best match (selected by lowest P-value)
          /motifs/{jaspar_name}/best_pvalue (attr) - P-value corresponding to best match
          /motifs/{jaspar_name}/best_query_id (attr) - Query ID of best match
          /motifs/{jaspar_name}/best_tissue_idx (attr) - Tissue index of best match
          /motifs/{jaspar_name}/best_query_ppm [L,4] - PPM of the best query motif
          /motifs/{jaspar_name}/tomtom_matches/* columns saved as datasets
          /motifs/{jaspar_name}/tissues/tissue_{t}/ ... aggregated seqlet-level metadata
            - example_idx [N]
            - start [N]
            - end [N]
            - is_revcomp [N] (optional)
            - pattern_group [N] (as bytes)
            - pattern_index [N]
            - contrib_score [L,D] (motif-level, same for all seqlets in tissue-motif group)
            - hypothetical_contribs [L,D] (motif-level, same for all seqlets in tissue-motif group)
            - saliency statistics as attributes (P05, P50, P95, n_seqlets)
          /analysis/cross_tissue_saliency/* cross-tissue saliency statistics
          /analysis/wilcoxon_tests/* Wilcoxon rank-sum test results
          /analysis/top_hits/* tissue-specific top hit motifs
        """
        os.makedirs(output_dir, exist_ok=True)
        h5_path = os.path.join(output_dir, output_name)
        print(f"Saving motif clusters H5 to {h5_path} ...")
        
        if not hasattr(self, 'merged_seqlets_df') or self.merged_seqlets_df is None or len(self.merged_seqlets_df) == 0:
            print("Warning: No merged seqlets DataFrame found. Skipping H5 save.")
            return
        
        with h5py.File(h5_path, "w") as h:
            motifs_grp = h.create_group("motifs")
            
            # Group by jaspar_name to iterate over motifs
            for jaspar_name, motif_seqlets in self.merged_seqlets_df.groupby("jaspar_name"):
                mgrp = motifs_grp.create_group(str(jaspar_name))
                
                # Store target PPM and ID from motif metadata
                if hasattr(self, 'motif_metadata') and jaspar_name in self.motif_metadata:
                    metadata = self.motif_metadata[jaspar_name]
                    if metadata.get("target_ppm") is not None:
                        ppm = np.asarray(metadata["target_ppm"], dtype=np.float32)
                        mgrp.create_dataset("target_ppm", data=ppm, compression="gzip")
                    if metadata.get("target_id") is not None:
                        mgrp.attrs["target_id"] = str(metadata["target_id"])
                    
                    # Store best match information
                    if metadata.get("best_evalue") is not None:
                        mgrp.attrs["best_evalue"] = metadata["best_evalue"]
                    if metadata.get("best_pvalue") is not None:
                        mgrp.attrs["best_pvalue"] = metadata["best_pvalue"]
                    if metadata.get("best_query_id") is not None:
                        mgrp.attrs["best_query_id"] = str(metadata["best_query_id"])
                    if metadata.get("best_tissue_idx") is not None:
                        mgrp.attrs["best_tissue_idx"] = metadata["best_tissue_idx"]
                    
                    # Store best query PPM
                    if metadata.get("best_query_ppm") is not None:
                        query_ppm = np.asarray(metadata["best_query_ppm"], dtype=np.float32)
                        mgrp.create_dataset("best_query_ppm", data=query_ppm, compression="gzip")
                
                # Store matches table subsets (from tomtom_matches filtered for this jaspar)
                if self.tomtom_matches is not None and len(self.tomtom_matches) > 0:
                    sub = self.tomtom_matches[self.tomtom_matches["Jaspar_Name"] == jaspar_name]
                    if len(sub) > 0:
                        match_grp = mgrp.create_group("tomtom_matches")
                        def _to_bytes(arr):
                            return np.asarray([str(x).encode("utf-8") for x in arr])
                        for col in [
                            "Tissue_Index", "Query_ID", "Pattern_Group", "Pattern_Index",
                            "Target_ID", "Jaspar_Name", "rc", "P_value", "E_value", "Q_value", "Offset", "Aligned_Length"
                        ]:
                            if col in sub.columns:
                                vals = sub[col].values
                                if col in ("Query_ID", "Pattern_Group", "Target_ID", "Jaspar_Name"):
                                    match_grp.create_dataset(col, data=_to_bytes(vals), compression="gzip")
                                elif col == "rc":
                                    match_grp.create_dataset(col, data=vals.astype(bool))
                                elif col == "Offset":
                                    match_grp.create_dataset(col, data=vals.astype(np.int32))
                                elif col == "Aligned_Length":
                                    match_grp.create_dataset(col, data=vals.astype(np.int32))
                                else:
                                    match_grp.create_dataset(col, data=vals)
                
                # Per-tissue aggregation using DataFrame groupby
                tissues_grp = mgrp.create_group("tissues")
                for tissue_idx, tissue_seqlets in motif_seqlets.groupby("tissue_idx"):
                    if len(tissue_seqlets) == 0:
                        continue
                    
                    tgrp = tissues_grp.create_group(f"tissue_{tissue_idx}")
                    tgrp.attrs["n_seqlets"] = len(tissue_seqlets)
                    
                    # Basic fields from DataFrame columns
                    example_idx = tissue_seqlets["example_idx"].values.astype(np.int32)
                    starts = tissue_seqlets["start"].values.astype(np.int32)
                    ends = tissue_seqlets["end"].values.astype(np.int32)
                    pat_grp = np.asarray([str(x).encode("utf-8") for x in tissue_seqlets["pattern_group"].values])
                    pat_idx = tissue_seqlets["pattern_index"].values.astype(np.int32)
                    
                    tgrp.create_dataset("example_idx", data=example_idx)
                    tgrp.create_dataset("start", data=starts)
                    tgrp.create_dataset("end", data=ends)
                    tgrp.create_dataset("pattern_group", data=pat_grp)
                    tgrp.create_dataset("pattern_index", data=pat_idx)
                    
                    # Save absolute coordinates if available
                    if "abs_start" in tissue_seqlets.columns and tissue_seqlets["abs_start"].notna().any():
                        abs_starts = tissue_seqlets["abs_start"].values.astype(np.int32)
                        abs_ends = tissue_seqlets["abs_end"].values.astype(np.int32)
                        tgrp.create_dataset("abs_start", data=abs_starts)
                        tgrp.create_dataset("abs_end", data=abs_ends)
                    
                    # Optional is_revcomp
                    if "is_revcomp" in tissue_seqlets.columns and tissue_seqlets["is_revcomp"].notna().any():
                        is_rev = tissue_seqlets["is_revcomp"].fillna(False).values.astype(np.bool_)
                        tgrp.create_dataset("is_revcomp", data=is_rev)
                    
                    # Handle contrib_score arrays (same for all seqlets in this tissue-motif group)
                    if "contrib_score" in tissue_seqlets.columns and tissue_seqlets["contrib_score"].notna().any():
                        # Take the first non-null contrib_score since they're all identical
                        contrib_score = tissue_seqlets["contrib_score"].dropna().iloc[0] if tissue_seqlets["contrib_score"].dropna().size > 0 else None
                        if contrib_score is not None:
                            arr = np.asarray(contrib_score)
                            if arr.ndim == 3:
                                # some formats are [1, L, D]
                                arr = arr[0]
                            tgrp.create_dataset("contrib_score", data=arr.astype(np.float32), compression="gzip")
                    
                    # Handle hypothetical_contribs arrays (same for all seqlets in this tissue-motif group)
                    if "hypothetical_contribs" in tissue_seqlets.columns and tissue_seqlets["hypothetical_contribs"].notna().any():
                        # Take the first non-null hypothetical_contribs since they're all identical
                        hyp_contrib = tissue_seqlets["hypothetical_contribs"].dropna().iloc[0] if tissue_seqlets["hypothetical_contribs"].dropna().size > 0 else None
                        if hyp_contrib is not None:
                            arr = np.asarray(hyp_contrib)
                            if arr.ndim == 3:
                                arr = arr[0]
                            tgrp.create_dataset("hypothetical_contribs", data=arr.astype(np.float32), compression="gzip")
                    
                    # grads_saliency: use pre-computed saliency scores from cross_tissue computation
                    # This avoids re-reading grads files during save
                    if hasattr(self, 'cross_tissue_stats_df') and self.cross_tissue_stats_df is not None:
                        # Extract saliency scores for this tissue-motif combination
                        motif_tissue_stats = self.cross_tissue_stats_df[
                            (self.cross_tissue_stats_df['Jaspar_Name'] == jaspar_name) & 
                            (self.cross_tissue_stats_df['Tissue'] == tissue_idx)
                        ]
                        if len(motif_tissue_stats) > 0:
                            # Store summary statistics instead of raw grads
                            tgrp.attrs["saliency_p05"] = float(motif_tissue_stats.iloc[0]["P05"])
                            tgrp.attrs["saliency_p50"] = float(motif_tissue_stats.iloc[0]["P50"])
                            tgrp.attrs["saliency_p95"] = float(motif_tissue_stats.iloc[0]["P95"])
                            # Store the total seqlets across all tissues for this motif
                            tgrp.attrs["saliency_total_seqlets"] = int(motif_tissue_stats.iloc[0]["N"])
                    
                    # Store the actual seqlets count for this tissue
                    tgrp.attrs["saliency_n_seqlets"] = len(tissue_seqlets)
                    
                    # Note: Raw grads_saliency arrays are not saved to avoid I/O bottleneck
                    # Use cross_tissue_stats_df for detailed saliency analysis
            
            # Save analysis results
            analysis_grp = h.create_group("analysis")
            
            # Save cross-tissue saliency stats if available
            if hasattr(self, 'cross_tissue_stats_df') and self.cross_tissue_stats_df is not None and len(self.cross_tissue_stats_df) > 0:
                ct_grp = analysis_grp.create_group("cross_tissue_saliency")
                
                def _to_bytes_array(arr):
                    return np.asarray([str(x).encode("utf-8") for x in arr])
                
                for col in self.cross_tissue_stats_df.columns:
                    vals = self.cross_tissue_stats_df[col].values
                    if col in ["Jaspar_Name", "Tissue"]:
                        ct_grp.create_dataset(col, data=_to_bytes_array(vals), compression="gzip")
                    else:
                        ct_grp.create_dataset(col, data=vals)

            # Save Wilcoxon rank-sum results if available
            if hasattr(self, 'wilcoxon_results_df') and self.wilcoxon_results_df is not None and len(self.wilcoxon_results_df) > 0:
                wx_grp = analysis_grp.create_group("wilcoxon_tests")
                
                def _to_bytes_array(arr):
                    return np.asarray([str(x).encode("utf-8") for x in arr])
                
                for col in self.wilcoxon_results_df.columns:
                    vals = self.wilcoxon_results_df[col].values
                    if col in ["Jaspar_Name", "Top_Tissue", "Second_Tissue"]:
                        wx_grp.create_dataset(col, data=_to_bytes_array(vals), compression="gzip")
                    else:
                        wx_grp.create_dataset(col, data=vals)

            # Save top hits if available
            if hasattr(self, 'top_hits_df') and self.top_hits_df is not None and len(self.top_hits_df) > 0:
                th_grp = analysis_grp.create_group("top_hits")
                
                def _to_bytes_array(arr):
                    return np.asarray([str(x).encode("utf-8") for x in arr])
                
                for col in self.top_hits_df.columns:
                    vals = self.top_hits_df[col].values
                    if col in ["Jaspar_Name", "Top_Tissue"]:
                        th_grp.create_dataset(col, data=_to_bytes_array(vals), compression="gzip")
                    else:
                        th_grp.create_dataset(col, data=vals)
            
            # Save individual seqlet saliency scores if available
            if hasattr(self, 'seqlet_saliency_df') and self.seqlet_saliency_df is not None and len(self.seqlet_saliency_df) > 0:
                ss_grp = analysis_grp.create_group("seqlet_saliency_scores")
                
                def _to_bytes_array(arr):
                    return np.asarray([str(x).encode("utf-8") for x in arr])
                
                for col in self.seqlet_saliency_df.columns:
                    vals = self.seqlet_saliency_df[col].values
                    if col in ["Jaspar_Name"]:
                        ss_grp.create_dataset(col, data=_to_bytes_array(vals), compression="gzip")
                    else:
                        ss_grp.create_dataset(col, data=vals)
            
            # Note: tissue_saliency_scores is deprecated in favor of cross_tissue_stats_df
            # The old structure is no longer used in the new pipeline
        
        print("Motif clusters H5 saved.")
    
    def compute_cross_tissue_saliency_distributions(self) -> pd.DataFrame:
        """For each motif cluster, compute per-tissue saliency distributions over the same set of seqlets.
        Uses mean over [length, channels] of grads slices from each tissue's grads file.
        Returns a long-form DataFrame with per-motif, per-tissue percentiles and counts.
        """
        print("Computing cross-tissue saliency distributions...")
        if not hasattr(self, 'merged_seqlets_df') or self.merged_seqlets_df is None or len(self.merged_seqlets_df) == 0:
            raise ValueError("Must merge seqlet coordinates first")

        # Cache grads per tissue to avoid reopening files repeatedly
        grads_cache = {}
        print("Loading grads files into memory...")
        
        def get_tissue_grads(tissue_idx: int):
            if tissue_idx in grads_cache:
                return grads_cache[tissue_idx]
            grads_file = f"{self.grads_dir}/tissue_{tissue_idx}.h5"
            if not os.path.exists(grads_file):
                print(f"Warning: Grads file not found: {grads_file}")
                grads_cache[tissue_idx] = None
                return None
            try:
                with h5py.File(grads_file, "r") as f:
                    grads_cache[tissue_idx] = f["grads_saliency"][:]
                return grads_cache[tissue_idx]
            except Exception as e:
                print(f"Error loading grads file {grads_file}: {e}")
                grads_cache[tissue_idx] = None
                return None

        # Pre-load all needed grads files
        for tissue_idx in self.tissue_indices:
            get_tissue_grads(int(tissue_idx))

        records = []
        wilcoxon_rows = []
        # Store individual seqlet saliency scores for plotting
        seqlet_saliency_records = []

        total_motifs = self.merged_seqlets_df['jaspar_name'].nunique()
        print(f"Processing {total_motifs} motifs...")
        
        for motif_idx, (jaspar_name, motif_seqlets) in enumerate(self.merged_seqlets_df.groupby("jaspar_name")):
            if motif_idx % 10 == 0:  # Print progress every 10 motifs
                print(f"Processing motif {motif_idx + 1}/{total_motifs}: {jaspar_name}")
            
            # Build per-selected-tissue distributions by aggregating seqlets from ALL origin tissues
            # For each seqlet, read grads from its origin-tissue file and slice last dim by selected tissue
            tissue_to_scores = {int(ti): [] for ti in self.tissue_indices}

            for seqlet_idx, row in motif_seqlets.iterrows():
                try:
                    origin_tissue = int(row["tissue_idx"])  # grads N dimension belongs to this origin tissue
                    grads_arr = get_tissue_grads(origin_tissue)
                    if grads_arr is None:
                        continue
                    ex = int(row["example_idx"])
                    st = int(row["start"])
                    en = int(row["end"])
                    if en <= st:
                        continue
                    
                    # Convert coordinates if window_size is specified
                    grads_length = grads_arr.shape[1]  # Get sequence length from grads array
                    abs_st, abs_en = self.convert_seqlet_coordinates(st, en, grads_length)
                    
                    # Validate converted coordinates
                    if abs_en <= abs_st or abs_st < 0 or abs_en > grads_length:
                        continue
                    
                    for sel_t in self.tissue_indices:
                        sel_t = int(sel_t)
                        if sel_t >= grads_arr.shape[3]:
                            continue
                        window = grads_arr[ex, abs_st:abs_en, :, sel_t]
                        saliency_score = float(np.mean(window))
                        tissue_to_scores[sel_t].append(saliency_score)
                        
                        # Store individual seqlet saliency score for plotting
                        seqlet_saliency_records.append({
                            "Jaspar_Name": jaspar_name,
                            "Tissue": int(sel_t),
                            "Origin_Tissue": int(origin_tissue),
                            "Example_Idx": int(ex),
                            "Start": int(st),  # Keep original relative coordinates
                            "End": int(en),    # Keep original relative coordinates
                            "Abs_Start": int(abs_st),  # Add absolute coordinates
                            "Abs_End": int(abs_en),    # Add absolute coordinates
                            "Saliency_Score": saliency_score
                        })
                except Exception:
                    continue

            # Convert lists to arrays
            for t_idx in list(tissue_to_scores.keys()):
                tissue_to_scores[t_idx] = np.asarray(tissue_to_scores[t_idx], dtype=np.float32)

            # Summarize percentiles per tissue
            for t_idx, scores in tissue_to_scores.items():
                if scores.size == 0:
                    p05 = p50 = p95 = np.nan
                    nseq = 0
                else:
                    p05 = float(np.quantile(scores, 0.05))
                    p50 = float(np.quantile(scores, 0.50))
                    p95 = float(np.quantile(scores, 0.95))
                    nseq = int(scores.size)
                records.append({
                    "Jaspar_Name": jaspar_name,
                    "Tissue": int(t_idx),
                    "P05": p05,
                    "P50": p50,
                    "P95": p95,
                    "N": nseq
                })

            # Dual statistical testing approach: P95 branch (activation) and P05 branch (repression)
            if len(tissue_to_scores) >= 2:
                # Get target_id from motif_metadata
                target_id = None
                if hasattr(self, 'motif_metadata') and jaspar_name in self.motif_metadata:
                    target_id = self.motif_metadata[jaspar_name].get("target_id")
                
                # P95 branch: Find top two tissues by P95 (activation)
                t_stats_p95 = []
                for t_idx, scores in tissue_to_scores.items():
                    if scores.size == 0:
                        continue
                    t_stats_p95.append((t_idx, float(np.quantile(scores, 0.95))))
                
                if len(t_stats_p95) >= 2:
                    t_stats_p95.sort(key=lambda x: x[1], reverse=True)
                    top_t_p95, top_p95 = t_stats_p95[0]
                    sec_t_p95, sec_p95 = t_stats_p95[1]
                    x_p95 = tissue_to_scores[top_t_p95]
                    y_p95 = tissue_to_scores[sec_t_p95]
                    
                    # U test for P95 branch (alternative="greater")
                    p95_u_stat, p95_pval = np.nan, np.nan
                    if x_p95.size > 0 and y_p95.size > 0:
                        try:
                            p95_u_stat, p95_pval = mannwhitneyu(x_p95, y_p95, alternative='greater')
                        except Exception:
                            pass
                else:
                    top_t_p95, sec_t_p95, top_p95, sec_p95 = None, None, np.nan, np.nan
                    p95_u_stat, p95_pval = np.nan, np.nan
                
                # P05 branch: Find tissues with negative values and lowest P05 (repression)
                t_stats_p05 = []
                for t_idx, scores in tissue_to_scores.items():
                    if scores.size == 0:
                        continue
                    p05_val = float(np.quantile(scores, 0.05))
                    # Only consider tissues with negative P05 values for repression analysis
                    if p05_val < 0:
                        t_stats_p05.append((t_idx, p05_val))
                
                # Sort by P05 value (ascending, so most negative comes first)
                if len(t_stats_p05) >= 2:
                    t_stats_p05.sort(key=lambda x: x[1])
                    top_t_p05, top_p05 = t_stats_p05[0]  # Most negative P05
                    sec_t_p05, sec_p05 = t_stats_p05[1]  # Second most negative P05
                    x_p05 = tissue_to_scores[top_t_p05]
                    y_p05 = tissue_to_scores[sec_t_p05]
                    
                    # U test for P05 branch (alternative="less")
                    p05_u_stat, p05_pval = np.nan, np.nan
                    if x_p05.size > 0 and y_p05.size > 0:
                        try:
                            p05_u_stat, p05_pval = mannwhitneyu(x_p05, y_p05, alternative='less')
                        except Exception:
                            pass
                else:
                    top_t_p05, sec_t_p05, top_p05, sec_p05 = None, None, np.nan, np.nan
                    p05_u_stat, p05_pval = np.nan, np.nan
                
                # Store results for both branches
                wilcoxon_rows.append({
                    "Jaspar_Name": jaspar_name,
                    "Target_ID": target_id,
                    # P95 branch (activation)
                    "Top_Tissue_P95": int(top_t_p95) if top_t_p95 is not None else None,
                    "Second_Tissue_P95": int(sec_t_p95) if sec_t_p95 is not None else None,
                    "Top_P95": float(top_p95) if not pd.isna(top_p95) else None,
                    "Second_P95": float(sec_p95) if not pd.isna(sec_p95) else None,
                    "P95_U_stat": float(p95_u_stat) if not pd.isna(p95_u_stat) else None,
                    "P95_P_value": float(p95_pval) if not pd.isna(p95_pval) else None,
                    # P05 branch (repression)
                    "Top_Tissue_P05": int(top_t_p05) if top_t_p05 is not None else None,
                    "Second_Tissue_P05": int(sec_t_p05) if sec_t_p05 is not None else None,
                    "Top_P05": float(top_p05) if not pd.isna(top_p05) else None,
                    "Second_P05": float(sec_p05) if not pd.isna(sec_p05) else None,
                    "P05_U_stat": float(p05_u_stat) if not pd.isna(p05_u_stat) else None,
                    "P05_P_value": float(p05_pval) if not pd.isna(p05_pval) else None
                })

        self.cross_tissue_stats_df = pd.DataFrame.from_records(records)
        self.wilcoxon_results_df = pd.DataFrame.from_records(wilcoxon_rows)
        # Store individual seqlet saliency scores
        self.seqlet_saliency_df = pd.DataFrame.from_records(seqlet_saliency_records)
        print(f"Computed cross-tissue stats for {self.cross_tissue_stats_df['Jaspar_Name'].nunique() if len(self.cross_tissue_stats_df)>0 else 0} motifs")
        print(f"Stored {len(self.seqlet_saliency_df)} individual seqlet saliency scores for plotting")
        return self.cross_tissue_stats_df

    def select_top_hits(self, alpha: float = 0.05) -> pd.DataFrame:
        """Select top-hit motifs with dual testing approach for activation and repression.
        
        Logic:
        - P95 branch: Test if top tissue's P95 is significantly higher than second (activation)
        - P05 branch: Test if top-P05 tissue's P05 is significantly lower than second-P05 (repression)
        - Any significant test qualifies as a top hit
        - Polarity annotation: "activation", "repression", or "both"
        
        E-value filtering is already done in parse_tomtom_results.
        Returns a DataFrame of top hits with polarity information.
        """
        print("Selecting top-hit motifs with dual testing approach...")
        if not hasattr(self, 'wilcoxon_results_df') or self.wilcoxon_results_df is None or len(self.wilcoxon_results_df) == 0:
            raise ValueError("Wilcoxon results not available. Run compute_cross_tissue_saliency_distributions first.")

        hits = []
        for _, row in self.wilcoxon_results_df.iterrows():
            jaspar = row["Jaspar_Name"]
            
            # Check P95 branch significance (activation)
            p95_pval = row.get("P95_P_value")
            p95_significant = not pd.isna(p95_pval) and float(p95_pval) <= alpha
            
            # Check P05 branch significance (repression)
            p05_pval = row.get("P05_P_value")
            p05_significant = not pd.isna(p05_pval) and float(p05_pval) <= alpha
            
            # Only include if at least one branch is significant
            if not (p95_significant or p05_significant):
                continue
            
            # Determine polarity
            if p95_significant and p05_significant:
                polarity = "both"
                # Use P95 branch as primary for "both" cases
                top_tissue = int(row["Top_Tissue_P95"]) if not pd.isna(row["Top_Tissue_P95"]) else None
                primary_pval = float(p95_pval)
            elif p95_significant:
                polarity = "activation"
                top_tissue = int(row["Top_Tissue_P95"]) if not pd.isna(row["Top_Tissue_P95"]) else None
                primary_pval = float(p95_pval)
            else:  # p05_significant
                polarity = "repression"
                top_tissue = int(row["Top_Tissue_P05"]) if not pd.isna(row["Top_Tissue_P05"]) else None
                primary_pval = float(p05_pval)
            
            # Get motif metadata
            target_id = None
            best_evalue = None
            best_pvalue = None
            direction = None
            if hasattr(self, 'motif_metadata') and jaspar in self.motif_metadata:
                metadata = self.motif_metadata[jaspar]
                target_id = metadata.get("target_id")
                best_evalue = metadata.get("best_evalue")
                best_pvalue = metadata.get("best_pvalue")
                # Get direction from tomtom_matches for the best match
                if hasattr(self, 'tomtom_matches') and self.tomtom_matches is not None:
                    best_match = self.tomtom_matches[
                        (self.tomtom_matches["Jaspar_Name"] == jaspar) & 
                        (self.tomtom_matches["P_value"] == best_pvalue)
                    ]
                    if len(best_match) > 0:
                        direction = "reverse" if best_match.iloc[0]["rc"] else "forward"
            
            # Build hit record
            hit_record = {
                "Jaspar_Name": jaspar,
                "Target_ID": target_id,
                "Best_Evalue": best_evalue,
                "Direction": direction,
                "Top_Tissue": top_tissue,
                "Polarity": polarity,
                "Primary_P_value": primary_pval,
                # P95 branch results
                "P95_Significant": p95_significant,
                "P95_P_value": float(p95_pval) if not pd.isna(p95_pval) else None,
                "Top_Tissue_P95": int(row["Top_Tissue_P95"]) if not pd.isna(row["Top_Tissue_P95"]) else None,
                "Second_Tissue_P95": int(row["Second_Tissue_P95"]) if not pd.isna(row["Second_Tissue_P95"]) else None,
                "Top_P95": float(row["Top_P95"]) if not pd.isna(row["Top_P95"]) else None,
                "Second_P95": float(row["Second_P95"]) if not pd.isna(row["Second_P95"]) else None,
                # P05 branch results
                "P05_Significant": p05_significant,
                "P05_P_value": float(p05_pval) if not pd.isna(p05_pval) else None,
                "Top_Tissue_P05": int(row["Top_Tissue_P05"]) if not pd.isna(row["Top_Tissue_P05"]) else None,
                "Second_Tissue_P05": int(row["Second_Tissue_P05"]) if not pd.isna(row["Second_Tissue_P05"]) else None,
                "Top_P05": float(row["Top_P05"]) if not pd.isna(row["Top_P05"]) else None,
                "Second_P05": float(row["Second_P05"]) if not pd.isna(row["Second_P05"]) else None
            }
            
            hits.append(hit_record)

        self.top_hits_df = pd.DataFrame.from_records(hits)
        
        # Print summary statistics
        if len(self.top_hits_df) > 0:
            polarity_counts = self.top_hits_df['Polarity'].value_counts()
            print(f"Selected {len(self.top_hits_df)} top-hit motifs:")
            for polarity, count in polarity_counts.items():
                print(f"  - {polarity}: {count}")
        else:
            print("No significant top-hit motifs found")
        
        return self.top_hits_df
    
    def save_results(self, output_dir: str = None):
        """Save all results to files.
        
        Args:
            output_dir: Output directory path
        """
        if output_dir is None:
            output_dir = self.output_dir
        print(f"Saving results to {output_dir}...")
        
        # Save Tomtom matches (always save as it's not in H5)
        if self.tomtom_matches is not None:
            self.tomtom_matches.to_csv(f"{output_dir}/tomtom_matches.tsv", sep="\t", index=False)
        
        # Save merged seqlets DataFrame as TSV
        if hasattr(self, 'merged_seqlets_df') and self.merged_seqlets_df is not None:
            # Create a copy for saving, excluding problematic array columns
            df_to_save = self.merged_seqlets_df.copy()
            
            # Remove array columns that cause TSV formatting issues
            problematic_columns = ['contrib_score', 'hypothetical_contribs', 'sequence']
            for col in problematic_columns:
                if col in df_to_save.columns:
                    df_to_save = df_to_save.drop(columns=[col])
            
            df_to_save.to_csv(f"{output_dir}/merged_seqlets.tsv", sep="\t", index=False)
            print(f"Saved merged seqlets to {output_dir}/merged_seqlets.tsv (excluded array columns)")
        
        # Save comprehensive H5 file with all data
        if hasattr(self, 'merged_seqlets_df') and self.merged_seqlets_df is not None:
            self.save_motif_clusters_h5(output_dir)
        
        # Save analysis results as TSV files
        if hasattr(self, 'cross_tissue_stats_df') and self.cross_tissue_stats_df is not None:
            self.cross_tissue_stats_df.to_csv(f"{output_dir}/cross_tissue_saliency_stats.tsv", sep="\t", index=False)
            print(f"Saved cross-tissue stats to {output_dir}/cross_tissue_saliency_stats.tsv")
        
        if hasattr(self, 'wilcoxon_results_df') and self.wilcoxon_results_df is not None:
            self.wilcoxon_results_df.to_csv(f"{output_dir}/wilcoxon_results.tsv", sep="\t", index=False)
            print(f"Saved Wilcoxon results to {output_dir}/wilcoxon_results.tsv")
        
        if hasattr(self, 'top_hits_df') and self.top_hits_df is not None:
            self.top_hits_df.to_csv(f"{output_dir}/top_hits.tsv", sep="\t", index=False)
            print(f"Saved top hits to {output_dir}/top_hits.tsv")
        
        # Save individual seqlet saliency scores if available
        if hasattr(self, 'seqlet_saliency_df') and self.seqlet_saliency_df is not None:
            self.seqlet_saliency_df.to_csv(f"{output_dir}/seqlet_saliency_scores.tsv", sep="\t", index=False)
            print(f"Saved seqlet saliency scores to {output_dir}/seqlet_saliency_scores.tsv")
        
        print("Results saved successfully!")
    

    
    def run_pipeline(self):
        """Run the complete motif analysis pipeline."""
        print("Starting Motif Analysis Pipeline...")
        
        # Step 1: Parse Tomtom results
        self.parse_tomtom_results()
        
        # Step 2: Merge seqlet coordinates
        self.merge_seqlet_coordinates()
        
        # Step 3: Cross-tissue saliency distributions and tests
        self.compute_cross_tissue_saliency_distributions()
        self.select_top_hits()
        
        # Step 4: Save results
        self.save_results()
        
        print("Pipeline completed successfully!")
        
        return {
            "tomtom_matches": self.tomtom_matches,
            "merged_seqlets_df": self.merged_seqlets_df,
            "cross_tissue_stats": getattr(self, 'cross_tissue_stats_df', None),
            "wilcoxon_results": getattr(self, 'wilcoxon_results_df', None),
            "top_hits": getattr(self, 'top_hits_df', None)
        }


def main():
    parser = argparse.ArgumentParser(description="Run motif analysis pipeline")
    parser.add_argument("-t", "--tissues", dest="tissue_indices", required=True, help="Tissues to analyze, comma-separated list of tissue indices")
    parser.add_argument("--grads_dir", required=True, help="Raw gradients directory name")
    parser.add_argument("--data_dir", required=True, help="modisco results, tomtom results directory path")
    parser.add_argument("--evalue_thresh", type=float, default=0.1, help="E-value threshold for Tomtom matches")
    parser.add_argument("--output_dir", default="motif_analysis_results", help="Output directory")
    parser.add_argument("--window_size", type=int, default=None, help="Window size used in modiscolite (for coordinate conversion)")
    parser.add_argument("--up_size", type=int, default=None, help="Upstream size used in preprocessing (for coordinate conversion from output window)")
    parser.add_argument("--down_size", type=int, default=None, help="Downstream size used in preprocessing (for coordinate conversion from output window)")

    
    
    args = parser.parse_args()
    
    # Create pipeline and run
    pipeline = MotifAnalysisPipeline(
        args.tissue_indices, args.grads_dir, args.data_dir, 
        args.output_dir, args.evalue_thresh, args.window_size, args.up_size, args.down_size)
    results = pipeline.run_pipeline()
    
    print(f"\nPipeline Summary:")
    print(f"- Total motif matches: {len(results['tomtom_matches'])}")
    print(f"- Unique motifs: {results['tomtom_matches']['Jaspar_Name'].nunique()}")
    print(f"- Total patterns processed: {results['tomtom_matches']['Query_ID'].nunique()}")
    print(f"- Cross-tissue saliency stats: {len(results['cross_tissue_stats']) if results['cross_tissue_stats'] is not None else 0}")
    print(f"- Wilcoxon tests performed: {len(results['wilcoxon_results']) if results['wilcoxon_results'] is not None else 0}")
    print(f"- Top hits selected: {len(results['top_hits']) if results['top_hits'] is not None else 0}")
    



if __name__ == "__main__":
    # debug mode
    import sys
    if len(sys.argv) == 1:
        sys.argv += [
            "-t", "0,1,14,18",
            "--data_dir", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/tfm_5tissues_atg_grads",
            "--grads_dir", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/5tissues_atg_grads",
            "--output_dir", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/modisco_analysis_results",
            "--evalue_thresh", "0.1",
            "--window_size", "32768",  # Center size used in preprocessing
            "--up_size", "2000",       # Example: 2000bp upstream
            "--down_size", "0",        # Example: 0bp downstream (user's case)
        ]
    main()