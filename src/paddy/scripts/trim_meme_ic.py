#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import math
from typing import List, Tuple
import re
import numpy as np
def parse_args():
    ap = argparse.ArgumentParser(description="Trim MEME motifs by information content (IC) > threshold from both ends.")
    ap.add_argument("input", help="Input MEME file (PPM: letter-probability matrix).")
    ap.add_argument("output", help="Output MEME file (trimmed).")
    ap.add_argument("--ic-thresh", type=float, default=0.1, help="IC threshold for trimming (default: 0.1).")
    ap.add_argument("--min-width", type=int, default=1, help="Minimum width to keep after trimming; if result shorter and --drop-empty is False, keep the best-single column (default: 1).")
    ap.add_argument("--drop-empty", action="store_true", help="Drop motifs that have no columns above threshold (instead of keeping the best single column).")
    ap.add_argument("--gc-content", type=float, default=0.435408, help="Genome GC fraction to set background as [AT,GC,GC,AT] = [(1-GC)/2, GC/2, GC/2, (1-GC)/2] (Rice_Nip: 0.43, Human: 0.41).")
    return ap.parse_args()

def information_content(row: List[float], background: List[float] | None = None, pseudocount: float = 0.001) -> float:
    # Match ic_clip behavior: ic_total = sum(p * log2((p+pc)/(1+4pc))) - sum(b * log2(b))
    # where p is the PWM row (A,C,G,T) and b is background (default uniform)
    if background is None:
        background = [0.25, 0.25, 0.25, 0.25]
    p = np.asarray(row, dtype=float)
    b = np.asarray(background, dtype=float)
    smoothed = (p + pseudocount) / (1.0 + 4.0 * pseudocount)
    term_pwm = np.sum(p * (np.log(smoothed) / np.log(2.0)))
    term_bg = np.sum(b * (np.log(b) / np.log(2.0)))
    return float(term_pwm - term_bg)

def read_meme(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
    # Split header (before first 'MOTIF') and motif blocks
    header_lines = []
    motifs = []  # list of dicts: {name, alt, matrices: [(alength, w, nsites, E, matrix, header_line_idx)]}
    i = 0
    # Capture header
    while i < len(lines) and not lines[i].startswith("MOTIF"):
        header_lines.append(lines[i])
        i += 1
    # Parse motifs
    while i < len(lines):
        if not lines[i].startswith("MOTIF"):
            i += 1
            continue
        # MOTIF line
        motif_line = lines[i].strip()
        parts = motif_line.split()
        # 'MOTIF <name> [alt...]'
        name = parts[1] if len(parts) > 1 else f"motif_{len(motifs)}"
        alt = " ".join(parts[2:]) if len(parts) > 2 else ""
        i += 1
        matrices = []
        # There may be additional fields (URL, etc.) or comments; we parse until next MOTIF or EOF
        while i < len(lines) and not lines[i].startswith("MOTIF"):
            line = lines[i]
            if line.strip().startswith("letter-probability matrix"):
                # parse header like: letter-probability matrix: alength= 4 w= 6 nsites= 20 E= 0
                header = line.strip()
                alength = None; w = None; nsites = None; E = None

                pairs = dict(re.findall(r'(\w+)\s*=\s*([^\s]+)', header))
                if "alength" in pairs:
                    alength = int(pairs["alength"])
                if "w" in pairs:
                    w = int(pairs["w"])
                if "nsites" in pairs:
                    try:
                        nsites = float(pairs["nsites"])
                    except:
                        nsites = None
                if "E" in pairs:
                    try:
                        E = float(pairs["E"])
                    except:
                        E = None

                if alength is None or w is None:
                    raise ValueError(f"Cannot parse matrix header: {header}")

                # read next w lines as rows of probabilities
                mat = []
                i += 1
                for _ in range(w):
                    if i >= len(lines):
                        raise ValueError("Unexpected EOF while reading matrix rows.")
                    row = [float(x) for x in lines[i].strip().split()[:alength]]
                    if len(row) != alength:
                        raise ValueError(f"Row has {len(row)} columns but alength={alength}: {lines[i]}")
                    mat.append(row)
                    i += 1
                matrices.append({"alength": alength, "w": w, "nsites": nsites, "E": E, "matrix": mat, "header": header})
                continue
            else:
                i += 1
                # preserve other lines with motif? We'll store them as 'extras' attached to motif if needed.
                continue
        motifs.append({"name": name, "alt": alt, "matrices": matrices})
    return header_lines, motifs

def trim_matrix(mat: List[List[float]], ic_thresh: float, min_width: int, drop_empty: bool, background: List[float] | None = None) -> List[List[float]]:
    # mat is list of rows; we want columns -> IC by column
    # Convert to columns
    alength = len(mat[0])
    cols = [[mat[r][c] for r in range(len(mat))] for c in range(alength)]
    # But in MEME, each ROW is A/C/G/T probability for a POSITION,
    # i.e., mat has shape (w positions, 4 bases). We want per-position IC.
    # So IC should be computed per ROW, not per column. Correct that:
    ic_scores = [information_content(row, background=background) for row in mat]
    # find left/right bounds where IC > thresh
    left = 0
    while left < len(ic_scores) and ic_scores[left] <= ic_thresh:
        left += 1
    right = len(ic_scores) - 1
    while right >= 0 and ic_scores[right] <= ic_thresh:
        right -= 1
    if left > right:
        # no positions above threshold
        if drop_empty:
            return []  # signal to drop
        # keep the single best-IC column
        if len(ic_scores) == 0:
            return []
        best = max(range(len(ic_scores)), key=lambda i: ic_scores[i])
        return [mat[best]]
    # slice
    trimmed = mat[left:right+1]
    # enforce min_width if possible: if shorter, we can expand to include flanking low-IC cols until min_width
    need = max(0, min_width - len(trimmed))
    if need > 0:
        # expand symmetrically if possible
        extra_left = min(left, need // 2 + need % 2)
        extra_right = min(len(mat) - 1 - right, need - extra_left)
        left -= extra_left
        right += extra_right
        trimmed = mat[left:right+1]
    return trimmed

def write_meme(header_lines: List[str], motifs, out_path: str):
    with open(out_path, "w") as out:
        # Ensure header includes 'MEME version' etc.
        if not header_lines:
            out.write("MEME version 5.4.1\n\n")
            out.write("ALPHABET= ACGT\n\n")
            out.write("strands: + -\n\n")
            out.write("Background letter frequencies (from unknown source):\n")
            out.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
        else:
            for line in header_lines:
                out.write(line)
        for m in motifs:
            out.write(f"MOTIF {m['name']}\n")
            for mat in m["matrices"]:
                alength = mat["alength"]
                w = len(mat["matrix"])
                nsites = mat["nsites"]
                E = mat["E"]
                out.write(f"letter-probability matrix: alength= {alength} w= {w}")
                if nsites is not None:
                    # MEME allows float nsites; print with sensible precision
                    out.write(f" nsites= {nsites:g}")
                if E is not None:
                    out.write(f" E= {E}")
                out.write("\n")
                for row in mat["matrix"]:
                    out.write(" ".join(f"{x:.6f}" for x in row) + "\n")
            out.write("\n")

def main():
    args = parse_args()
    header, motifs = read_meme(args.input)
    # Background per explore_modisco_4_tissues.py: AT=(1-GC)/2, GC=GC/2; order [A, C, G, T]
    at_pct = (1.0 - args.gc_content) / 2.0
    gc_pct = args.gc_content / 2.0
    background = [at_pct, gc_pct, gc_pct, at_pct]
    trimmed_motifs = []
    dropped = 0
    for m in motifs:
        new_mats = []
        for mat in m["matrices"]:
            if mat["alength"] != 4:
                raise ValueError(f"Only alength=4 supported (found {mat['alength']}) in motif {m['name']}")
            trimmed = trim_matrix(mat["matrix"], args.ic_thresh, args.min_width, args.drop_empty, background=background)
            if len(trimmed) == 0:
                continue
            new_mats.append({
                "alength": mat["alength"],
                "w": len(trimmed),
                "nsites": mat["nsites"],
                "E": mat["E"],
                "matrix": trimmed,
                "header": mat["header"],
            })
        if new_mats:
            trimmed_motifs.append({"name": m["name"], "alt": m["alt"], "matrices": new_mats})
        else:
            dropped += 1
    write_meme(header, trimmed_motifs, args.output)
    print(f"Done. Motifs in: {len(motifs)}; motifs out: {len(trimmed_motifs)}; dropped: {dropped}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv += [
            "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/modisco_results/tissue_1_motifs.meme",
            "/data/xhhuanglab/gulei/projects/paddy/motif_pipeline/tmp/tmp.meme"
        ]
    main()
