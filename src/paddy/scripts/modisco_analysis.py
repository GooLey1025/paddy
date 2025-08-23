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
    def __init__(self, tissue_indices: List[int], grads_dir: str, data_dir: str, output_dir: str, evalue_thresh: float = 0.1, tissues_dict: str = None):
        self.tissue_indices = [int(ti) for ti in tissue_indices.split(",")]
        self.grads_dir = grads_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.evalue_thresh = evalue_thresh
        self.tissues_dict = tissues_dict
        # Discover tomtom XML files under the provided directory
        self.tomtom_xml_files = []
        for ti in self.tissue_indices:
            tissue_dir = os.path.join(self.data_dir, f"tissue_{ti}")
            tomtom_file = os.path.join(tissue_dir, "tomtom.xml")
            if os.path.isdir(tissue_dir) and os.path.exists(tomtom_file):
                self.tomtom_xml_files.append(tomtom_file)
        
        # Tissue name mapping
        if tissues_dict is not None:
            with open(tissues_dict, "r") as f:
                self.tissue_names = json.load(f)
        else:
            self.tissue_names = {ti: f"Tissue_{ti}" for ti in self.tissue_indices}
        
        # Data storage
        self.tomtom_matches = None
        self.merged_seqlets = None
        self.tissue_saliency_scores = None
    
        os.makedirs(self.output_dir, exist_ok=True)

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
                    
                    if ev <= self.evalue_thresh:
                        all_rows.append({
                            "Tissue_Index": tissue_idx,
                            "Query_ID": q_id,
                            "Pattern_Group": pattern_group,
                            "Pattern_Index": pattern_index,
                            "Target_ID": target_motifs[t_idx]["id"],
                            "Jaspar_Name": target_motifs[t_idx]["alt"],
                            "rc": (t.attrib["rc"] == "y"),
                            "P_value": pv,
                            "E_value": ev,
                            "Q_value": qv,
                            "Query_PWM": query_motifs[q_idx]["ppm"],
                            "Target_PWM": target_motifs[t_idx]["ppm"],
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
            # Find the best match (lowest E-value) for this motif
            best_match = motif_group.loc[motif_group["E_value"].idxmin()]
            
            # Store motif metadata from the best match
            motif_metadata[jaspar_name] = {
                "target_pwm": np.asarray(best_match["Target_PWM"]) if isinstance(best_match["Target_PWM"], list) else None,
                "target_id": best_match["Target_ID"],
                "best_evalue": float(best_match["E_value"]),
                "best_pvalue": float(best_match["P_value"]),
                "best_query_id": best_match["Query_ID"]
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
                
                modisco_file = f"{self.data_dir}/tissue_{tissue_idx}.lite.h5"
                
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
                        # Create seqlet records for DataFrame
                        for i in range(n_seqlets):
                            seqlet_record = {
                                "tissue_idx": tissue_idx,
                                "example_idx": int(example_idx[i]),
                                "start": int(start[i]),
                                "end": int(end[i]),
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
          /motifs/{jaspar_name}/target_pwm [L,4]
          /motifs/{jaspar_name}/target_id (attr)
          /motifs/{jaspar_name}/best_evalue (attr) - lowest E-value among all matches
          /motifs/{jaspar_name}/best_pvalue (attr) - P-value corresponding to best match
          /motifs/{jaspar_name}/best_query_id (attr) - Query ID of best match
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
                
                # Store target PWM and ID from motif metadata
                if hasattr(self, 'motif_metadata') and jaspar_name in self.motif_metadata:
                    metadata = self.motif_metadata[jaspar_name]
                    if metadata.get("target_pwm") is not None:
                        pwm = np.asarray(metadata["target_pwm"], dtype=np.float32)
                        mgrp.create_dataset("target_pwm", data=pwm, compression="gzip")
                    if metadata.get("target_id") is not None:
                        mgrp.attrs["target_id"] = str(metadata["target_id"])
                    
                    # Store best match information
                    if metadata.get("best_evalue") is not None:
                        mgrp.attrs["best_evalue"] = metadata["best_evalue"]
                    if metadata.get("best_pvalue") is not None:
                        mgrp.attrs["best_pvalue"] = metadata["best_pvalue"]
                    if metadata.get("best_query_id") is not None:
                        mgrp.attrs["best_query_id"] = str(metadata["best_query_id"])
                
                # Store matches table subsets (from tomtom_matches filtered for this jaspar)
                if self.tomtom_matches is not None and len(self.tomtom_matches) > 0:
                    sub = self.tomtom_matches[self.tomtom_matches["Jaspar_Name"] == jaspar_name]
                    if len(sub) > 0:
                        match_grp = mgrp.create_group("tomtom_matches")
                        def _to_bytes(arr):
                            return np.asarray([str(x).encode("utf-8") for x in arr])
                        for col in [
                            "Tissue_Index", "Query_ID", "Pattern_Group", "Pattern_Index",
                            "Target_ID", "Jaspar_Name", "rc", "P_value", "E_value", "Q_value"
                        ]:
                            if col in sub.columns:
                                vals = sub[col].values
                                if col in ("Query_ID", "Pattern_Group", "Target_ID", "Jaspar_Name"):
                                    match_grp.create_dataset(col, data=_to_bytes(vals), compression="gzip")
                                elif col == "rc":
                                    match_grp.create_dataset(col, data=vals.astype(bool))
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
                    arr = f["grads_saliency"][:]
                # Assume fixed layout [N, L, D, T]
                grads_cache[tissue_idx] = arr
                print(f"Loaded grads for tissue {tissue_idx}: {arr.shape}")
                return arr
            except Exception as e:
                print(f"Error reading grads from {grads_file}: {e}")
                grads_cache[tissue_idx] = None
                return None
        
        # Pre-load all needed grads files
        for tissue_idx in self.tissue_indices:
            get_tissue_grads(int(tissue_idx))

        records = []
        wilcoxon_rows = []

        total_motifs = self.merged_seqlets_df['jaspar_name'].nunique()
        print(f"Processing {total_motifs} motifs...")
        
        for motif_idx, (jaspar_name, motif_seqlets) in enumerate(self.merged_seqlets_df.groupby("jaspar_name")):
            if motif_idx % 10 == 0:  # Print progress every 10 motifs
                print(f"Processing motif {motif_idx + 1}/{total_motifs}: {jaspar_name}")
            
            # Build per-selected-tissue distributions by aggregating seqlets from ALL origin tissues
            # For each seqlet, read grads from its origin-tissue file and slice last dim by selected tissue
            tissue_to_scores = {int(ti): [] for ti in self.tissue_indices}

            for _, row in motif_seqlets.iterrows():
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
                    for sel_t in self.tissue_indices:
                        sel_t = int(sel_t)
                        if sel_t >= grads_arr.shape[3]:
                            continue
                        window = grads_arr[ex, st:en, :, sel_t]
                        tissue_to_scores[sel_t].append(float(np.mean(window)))
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

            # Wilcoxon rank-sum: top vs second by 95th percentile
            if len(tissue_to_scores) >= 2:
                # Determine top two by P95
                t_stats = []
                for t_idx, scores in tissue_to_scores.items():
                    if scores.size == 0:
                        continue
                    t_stats.append((t_idx, float(np.quantile(scores, 0.95))))
                if len(t_stats) >= 2:
                    t_stats.sort(key=lambda x: x[1], reverse=True)
                    top_t, top_p95 = t_stats[0]
                    sec_t, sec_p95 = t_stats[1]
                    x = tissue_to_scores[top_t]
                    y = tissue_to_scores[sec_t]
                    if x.size > 0 and y.size > 0:
                        try:
                            stat, pval = mannwhitneyu(x, y, alternative='greater')
                        except Exception:
                            stat, pval = np.nan, np.nan
                        wilcoxon_rows.append({
                            "Jaspar_Name": jaspar_name,
                            "Top_Tissue": int(top_t),
                            "Second_Tissue": int(sec_t),
                            "Top_P95": float(top_p95),
                            "Second_P95": float(sec_p95),
                            "U_stat": float(stat) if not isinstance(stat, float) else stat,
                            "P_value": float(pval) if not isinstance(pval, float) else pval
                        })

        self.cross_tissue_stats_df = pd.DataFrame.from_records(records)
        self.wilcoxon_results_df = pd.DataFrame.from_records(wilcoxon_rows)
        print(f"Computed cross-tissue stats for {self.cross_tissue_stats_df['Jaspar_Name'].nunique() if len(self.cross_tissue_stats_df)>0 else 0} motifs")
        return self.cross_tissue_stats_df

    def select_top_hits(self, alpha: float = 0.05) -> pd.DataFrame:
        """Select top-hit motifs per tissue where top tissue's P95 is significantly higher than second.
        E-value filtering is already done in parse_tomtom_results.
        Returns a DataFrame of top hits.
        """
        print("Selecting top-hit motifs...")
        if not hasattr(self, 'wilcoxon_results_df') or self.wilcoxon_results_df is None or len(self.wilcoxon_results_df) == 0:
            raise ValueError("Wilcoxon results not available. Run compute_cross_tissue_saliency_distributions first.")

        hits = []
        for _, row in self.wilcoxon_results_df.iterrows():
            jaspar = row["Jaspar_Name"]
            top_t = int(row["Top_Tissue"])
            pval = float(row["P_value"]) if not pd.isna(row["P_value"]) else 1.0
            # significance by Wilcoxon only (E-value already filtered)
            if pval <= alpha:
                hits.append({
                    "Jaspar_Name": jaspar,
                    "Top_Tissue": top_t,
                    "Wilcoxon_P": pval,
                    "Top_P95": float(row["Top_P95"]),
                    "Second_Tissue": int(row["Second_Tissue"]),
                    "Second_P95": float(row["Second_P95"])
                })

        self.top_hits_df = pd.DataFrame.from_records(hits)
        print(f"Selected {len(self.top_hits_df)} top-hit motifs")
        return self.top_hits_df
    
    def analyze_tissue_specificity(self) -> pd.DataFrame:
        """Analyze tissue specificity of motifs based on saliency score distributions."""
        print("Analyzing tissue specificity...")
        
        if self.tissue_saliency_scores is None:
            raise ValueError("Must compute saliency scores first")
        
        specificity_analysis = []
        
        for jaspar_name, motif_data in self.tissue_saliency_scores.items():
            tissues_with_data = list(motif_data["tissues"].keys())
            
            if len(tissues_with_data) < 2:
                continue
            
            # Compare distributions between tissues
            for i, tissue1 in enumerate(tissues_with_data):
                for tissue2 in tissues_with_data[i+1:]:
                    scores1 = motif_data["tissues"][tissue1]["saliency_scores"]
                    scores2 = motif_data["tissues"][tissue2]["saliency_scores"]
                    
                    if len(scores1) > 0 and len(scores2) > 0:
                        # Compute effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1) + 
                                            (len(scores2) - 1) * np.var(scores2)) / 
                                           (len(scores1) + len(scores2) - 2))
                        
                        if pooled_std > 0:
                            cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                        else:
                            cohens_d = 0.0
                        
                        # Determine which tissue has higher scores
                        if np.mean(scores1) > np.mean(scores2):
                            higher_tissue = tissue1
                            lower_tissue = tissue2
                            higher_mean = np.mean(scores1)
                            lower_mean = np.mean(scores2)
                        else:
                            higher_tissue = tissue2
                            lower_tissue = tissue1
                            higher_mean = np.mean(scores2)
                            lower_mean = np.mean(scores1)
                        
                        specificity_analysis.append({
                            "Jaspar_Name": jaspar_name,
                            "Tissue1": tissue1,
                            "Tissue2": tissue2,
                            "Tissue1_Mean": float(np.mean(scores1)),
                            "Tissue2_Mean": float(np.mean(scores2)),
                            "Higher_Tissue": higher_tissue,
                            "Lower_Tissue": lower_tissue,
                            "Higher_Mean": float(higher_mean),
                            "Lower_Mean": float(lower_mean),
                            "Effect_Size": float(cohens_d),
                            "Tissue1_n": len(scores1),
                            "Tissue2_n": len(scores2),
                            "Fold_Change": float(higher_mean / lower_mean) if lower_mean > 0 else float('inf')
                        })
        
        specificity_df = pd.DataFrame(specificity_analysis)
        
        # Add tissue names
        specificity_df["Tissue1_Name"] = specificity_df["Tissue1"].map(self.tissue_names)
        specificity_df["Tissue2_Name"] = specificity_df["Tissue2"].map(self.tissue_names)
        specificity_df["Higher_Tissue_Name"] = specificity_df["Higher_Tissue"].map(self.tissue_names)
        
        print(f"Analyzed tissue specificity for {len(specificity_df)} motif-tissue comparisons")
        return specificity_df
    
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
    parser.add_argument("--tissues_dict", default=None, help="Tissues dictionary file for mapping tissue indices to names")

    
    
    args = parser.parse_args()
    
    # Create pipeline and run
    pipeline = MotifAnalysisPipeline(
        args.tissue_indices, args.grads_dir, args.data_dir, 
        args.output_dir, args.evalue_thresh, args.tissues_dict)
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
            "--tissues_dict", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/23tissues_dict.json",
            "--output_dir", "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/modisco_analysis_results",
            "--evalue_thresh", "0.1",
        ]
    main()