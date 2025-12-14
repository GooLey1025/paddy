# gene.py
# Gene sequence extraction and manipulation for Harrow
# Standalone implementation (no dependency on paddy.gene)

import gzip
import numpy as np
import pysam
import warnings
from intervaltree import IntervalTree
from typing import Optional, Tuple, Union


def gtf_kv(s):
    """Convert the last gtf section of key/value pairs into a dict.

    Handles both GTF (space-separated) and GFF3 (= separated) formats.

    Parameters
    ----------
    s : str
        Key-value string from GTF/GFF file (9th column)

    Returns
    -------
    d : dict
        Dictionary of key-value pairs
    """
    d = {}

    # Split by semicolon to get individual key-value pairs
    a = s.split(";")

    for key_val in a:
        key_val = key_val.strip()
        if not key_val:
            continue

        # Check if this uses = separator (like ID=value) - GFF3 format
        eq_i = key_val.find("=")
        if eq_i != -1:
            # Split on first = only
            key = key_val[:eq_i].strip()
            val = key_val[eq_i + 1:].strip()

            # Remove quotes if present
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]

        else:
            # Fall back to space-separated format (like gene_id "value") - GTF format
            parts = key_val.split()
            if len(parts) >= 2:
                key = parts[0]
                val = " ".join(parts[1:])

                # Remove quotes if present
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
            else:
                continue

        d[key] = val

    return d


class Gene:
    """Class for managing genes in an isoform-agnostic way, taking
    the union of exons across isoforms."""

    def __init__(self, chrom, strand, kv, name=None):
        """
        Initialize Gene.

        Parameters
        ----------
        chrom : str
            Chromosome name
        strand : str
            '+' or '-'
        kv : dict
            Key-value attributes from GTF/GFF
        name : str, optional
            Gene name/symbol
        """
        self.chrom = chrom
        self.strand = strand
        self.kv = kv
        self.name = name
        self.exons = IntervalTree()
        self.cds_intervals = IntervalTree()

    def add_exon(self, start, end):
        """Add an exon interval.

        Parameters
        ----------
        start : int
            Start position (0-based)
        end : int
            End position (0-based, exclusive)
        """
        self.exons[start:end] = True

    def add_cds(self, start, end):
        """Add a CDS interval.

        Parameters
        ----------
        start : int
            Start position (0-based)
        end : int
            End position (0-based, exclusive)
        """
        self.cds_intervals[start:end] = True

    def get_exons(self):
        """Get sorted, merged exon intervals."""
        self.exons.merge_overlaps()
        return sorted(self.exons)

    def midpoint(self):
        """Get gene midpoint position (0-based)."""
        positions = []
        for exon in self.get_exons():
            positions += range(exon.begin, exon.end)
        midp = int(np.mean(positions))
        return midp

    def span(self):
        """Get gene span (0-based coordinates).

        Returns
        -------
        gene_start : int
            Start position (0-based)
        gene_end : int
            End position (0-based, exclusive)
        """
        exon_starts = [exon.begin for exon in self.exons]
        exon_ends = [exon.end for exon in self.exons]
        return min(exon_starts), max(exon_ends)

    def get_cds(self):
        """Get sorted, merged CDS intervals."""
        if not hasattr(self, 'cds_intervals'):
            self.cds_intervals = IntervalTree()
        self.cds_intervals.merge_overlaps()
        return sorted(self.cds_intervals)

    def cds_start(self):
        """Get the start position of the first CDS (ATG position for + strand).

        Returns
        -------
        cds_start : int
            CDS start position (0-based). Returns gene start if no CDS available.
        """
        cds_list = self.get_cds()
        if not cds_list:
            # If no CDS, fall back to gene start
            gene_start, _ = self.span()
            return gene_start
        return min(cds.begin for cds in cds_list)

    def cds_end(self):
        """Get the end position of the last CDS (ATG position for - strand).

        Returns
        -------
        cds_end : int
            CDS end position (0-based, inclusive). Returns gene end if no CDS available.
        """
        cds_list = self.get_cds()
        if not cds_list:
            _, gene_end = self.span()
            return gene_end
        # IntervalTree uses right-open intervals, so subtract 1 for inclusive end
        return max(cds.end for cds in cds_list) - 1


class Transcriptome:
    """Transcriptome manager for parsing GTF/GFF and extracting gene sequences.

    This class provides functionality to:
    1. Parse GTF/GFF files and build gene structures
    2. Extract gene sequences centered on ATG start codon
    3. Handle strand orientation and boundary conditions

    Examples
    --------
    >>> from harrow import gene
    >>> transcriptome = gene.Transcriptome("genome.fa", "genes.gff3")
    >>> 
    >>> # Get sequence centered on ATG
    >>> ipa1 = transcriptome.gene(gene_name="LOC_Os08g39890", 
    ...                            seq_length=32768, middle="atg")
    >>> print(ipa1.seq[16384:16387])  # Should be "ATG"
    >>> 
    >>> # Get absolute genomic coordinates
    >>> start, end = ipa1.abs_position
    >>> 
    >>> # Get relative coordinates (to ATG)
    >>> rel_start, rel_end = ipa1.rel_position  # e.g., (-16384, 16384)
    """

    def __init__(self, genome_fasta: str, genes_gff: str):
        """
        Initialize Transcriptome.

        Parameters
        ----------
        genome_fasta : str
            Path to genome FASTA file
        genes_gff : str
            Path to genes GFF/GTF file
        """
        self.genome_fasta = genome_fasta
        self.genes_gff = genes_gff

        # Load genome
        self.genome_open = pysam.Fastafile(genome_fasta)
        self.chrom_lengths = dict(zip(self.genome_open.references,
                                      self.genome_open.lengths))

        # Load genes from GTF/GFF
        self.genes = {}
        self._read_gtf(genes_gff)

        print(f"Loaded {len(self.genes)} genes from {genes_gff}")

    def _read_gtf(self, gtf_file: str):
        """Parse GTF/GFF file and build gene structures.

        Parameters
        ----------
        gtf_file : str
            Path to GTF/GFF file
        """
        if gtf_file[-3:] == ".gz":
            gtf_in = gzip.open(gtf_file, "rt")
        else:
            gtf_in = open(gtf_file)

        # Skip header lines
        line = gtf_in.readline()
        while line and line[0] == "#":
            line = gtf_in.readline()

        while line:
            a = line.split("\t")
            if len(a) >= 9 and a[2] in ["exon", "CDS"]:
                chrom = a[0]
                start = int(a[3])  # 1-based, inclusive
                end = int(a[4])    # 1-based, exclusive
                strand = a[6]
                kv = gtf_kv(a[8])

                # Extract gene_id - handle multiple formats
                gene_id = None
                if "gene_id" in kv:
                    gene_id = kv["gene_id"]
                elif "Parent" in kv:
                    # Prefer Parent over ID so CDS/exon entries group under their parent transcript/gene
                    gene_id = kv["Parent"]
                    # Remove isoform suffix if present
                    if "." in gene_id:
                        gene_id = gene_id.rsplit(".", 1)[0]
                elif "ID" in kv:
                    # GFF3 format
                    id_val = kv["ID"]
                    if ":" in id_val:
                        # Handle cases like "LOC_Os08g39850.1:exon_3"
                        gene_id = id_val.split(":")[0]
                    else:
                        gene_id = id_val

                    # Remove isoform suffix (e.g., .1, .2, .3) to get primary gene ID
                    if "." in gene_id:
                        gene_id = gene_id.rsplit(".", 1)[0]

                if gene_id is None:
                    # Skip if we can't determine gene_id
                    line = gtf_in.readline()
                    continue

                # Extract gene_name (optional)
                gene_name = None
                if "gene_name" in kv:
                    gene_name = kv["gene_name"]
                elif "Name" in kv:
                    gene_name = kv["Name"]
                elif "gene_symbol" in kv:
                    gene_name = kv["gene_symbol"]

                # Initialize gene if needed
                if gene_id not in self.genes:
                    self.genes[gene_id] = Gene(chrom, strand, kv, gene_name)

                # Add exon or CDS (convert to 0-based)
                if a[2] == "exon":
                    self.genes[gene_id].add_exon(start - 1, end)
                elif a[2] == "CDS":
                    self.genes[gene_id].add_cds(start - 1, end)

            line = gtf_in.readline()

        gtf_in.close()

    def gene(self, gene_name: str, seq_length: int = 32768,
             middle: str = "atg") -> 'GeneSequence':
        """
        Extract gene sequence centered on specified position.

        Parameters
        ----------
        gene_name : str
            Gene identifier (e.g., "LOC_Os08g39890")
        seq_length : int, optional
            Total sequence length to extract. Default is 32768 (32kb).
        middle : str, optional
            What to center the sequence on:
            - "atg": Center on ATG start codon (default)
            - "midpoint": Center on gene midpoint

        Returns
        -------
        GeneSequence
            Container with sequence and metadata

        Raises
        ------
        ValueError
            If gene not found, or if sequence would extend beyond chromosome

        Notes
        -----
        - For + strand genes: ATG is at cds_start
        - For - strand genes: ATG is at cds_end (needs RC to get 5'->3')
        - All coordinates in output are 1-based
        - The center position (ATG) is at index seq_length//2 (0-indexed in array)
        - For 32kb: ATG is at positions 16384:16387 (0-indexed), or 16385-16387 (1-based)
        """
        # Find gene
        if gene_name not in self.genes:
            raise ValueError(
                f"Gene '{gene_name}' not found in {self.genes_gff}")

        gene = self.genes[gene_name]

        # Validate chromosome
        if gene.chrom not in self.chrom_lengths:
            raise ValueError(f"Chromosome '{gene.chrom}' not found in genome")

        chrom_len = self.chrom_lengths[gene.chrom]

        # Get center position (0-based)
        if middle == "atg":
            if gene.strand == "+":
                center_pos = gene.cds_start()
            else:
                center_pos = gene.cds_end()

            if center_pos is None or center_pos < 0:
                raise ValueError(
                    f"Gene '{gene_name}' has no valid CDS/ATG position")
        elif middle == "midpoint":
            center_pos = gene.midpoint()
        else:
            raise ValueError(
                f"Unknown middle type: {middle}. Use 'atg' or 'midpoint'")

        # Calculate sequence window (1-based coordinates)
        half_len = seq_length // 2

        if gene.strand == "+":
            seq_start_1based = center_pos + 1 - half_len
            seq_end_1based = seq_start_1based + seq_length
        else:
            # For - strand, center_pos is the last position of ATG (in genomic coords)
            # We need to adjust to get ATG in the center after RC
            seq_start_1based = center_pos + 1 - half_len + 1
            seq_end_1based = seq_start_1based + seq_length

        # Validate boundaries
        if seq_start_1based < 1:
            raise ValueError(
                f"Gene '{gene_name}': Sequence window extends before chromosome start "
                f"(seq_start={seq_start_1based}). Try smaller seq_length.")

        if seq_end_1based > chrom_len:
            raise ValueError(
                f"Gene '{gene_name}': Sequence window extends beyond chromosome end "
                f"(seq_end={seq_end_1based}, chrom_len={chrom_len}). Try smaller seq_length.")

        # Extract sequence
        seq_1hot, atg_found = self._extract_sequence(
            gene.chrom, seq_start_1based, seq_end_1based,
            seq_length, gene.strand, gene_name
        )

        # Convert to DNA string
        seq = self._one_hot_to_dna(seq_1hot)

        # Get CDS boundaries (1-based)
        cds_start_0based = gene.cds_start() if gene.cds_start() is not None else -1
        cds_end_0based = gene.cds_end() if gene.cds_end() is not None else -1
        cds_start_1based = cds_start_0based + 1 if cds_start_0based >= 0 else -1
        cds_end_1based = cds_end_0based + 1 if cds_end_0based >= 0 else -1

        # Calculate relative positions (to ATG center)
        rel_start = -half_len
        rel_end = half_len

        return GeneSequence(
            seq=seq,
            seq_1hot=seq_1hot,
            gene_id=gene_name,
            # Inclusive end
            abs_position=(seq_start_1based, seq_end_1based - 1),
            rel_position=(rel_start, rel_end - 1),  # Inclusive end
            chr=gene.chrom,
            strand=gene.strand,
            atg_pos=center_pos + 1,  # Convert to 1-based
            atg_found=atg_found,
            cds_start=cds_start_1based,
            cds_end=cds_end_1based
        )

    def _extract_sequence(self, chrm: str, start_1based: int, end_1based: int,
                          seq_len: int, strand: str, gene_name: str) -> Tuple[np.ndarray, bool]:
        """
        Extract sequence and convert to one-hot encoding.

        Parameters
        ----------
        chrm : str
            Chromosome name
        start_1based : int
            Start position (1-based, inclusive)
        end_1based : int
            End position (1-based, exclusive)
        seq_len : int
            Expected sequence length
        strand : str
            '+' or '-'
        gene_name : str
            Gene identifier (for warnings)

        Returns
        -------
        seq_1hot : np.ndarray, shape=(seq_len, 4)
            One-hot encoded sequence
        atg_found : bool
            Whether ATG was found at expected center position
        """
        # Convert to 0-based for pysam
        start_0based = start_1based - 1
        end_0based = end_1based - 1

        # Extract sequence
        if start_0based < 0:
            # Pad with N's at the beginning
            seq_dna = "N" * (-start_0based) + \
                self.genome_open.fetch(chrm, 0, end_0based)
        else:
            seq_dna = self.genome_open.fetch(chrm, start_0based, end_0based)

        # Extend to full length if needed
        if len(seq_dna) < seq_len:
            warnings.warn(
                f"Gene '{gene_name}': Sequence shorter than expected "
                f"({len(seq_dna)} < {seq_len}), padding with N's"
            )
            seq_dna += "N" * (seq_len - len(seq_dna))

        # Reverse complement for - strand
        if strand == "-":
            seq_dna = self._reverse_complement(seq_dna)

        # One-hot encode
        seq_1hot = self._dna_1hot(seq_dna)

        # Check for ATG at center
        center_idx = seq_len // 2
        atg_found = seq_dna[center_idx:center_idx+3].upper() == "ATG"

        if not atg_found:
            warnings.warn(
                f"Gene '{gene_name}' ({chrm}:{start_1based}-{end_1based}, strand={strand}): "
                f"ATG not found at expected center position (index {center_idx}), "
                f"found '{seq_dna[center_idx:center_idx+3]}' instead"
            )

        return seq_1hot, atg_found

    def _dna_1hot(self, seq: str) -> np.ndarray:
        """
        Convert DNA sequence to one-hot encoding.

        Parameters
        ----------
        seq : str
            DNA sequence

        Returns
        -------
        seq_1hot : np.ndarray, shape=(len(seq), 4)
            One-hot encoded sequence [A, C, G, T]
        """
        seq = seq.upper()
        seq_len = len(seq)
        seq_1hot = np.zeros((seq_len, 4), dtype='float32')

        for i, nt in enumerate(seq):
            if nt == 'A':
                seq_1hot[i, 0] = 1
            elif nt == 'C':
                seq_1hot[i, 1] = 1
            elif nt == 'G':
                seq_1hot[i, 2] = 1
            elif nt == 'T':
                seq_1hot[i, 3] = 1
            # else: N or other ambiguous bases remain all zeros

        return seq_1hot

    def _one_hot_to_dna(self, seq_1hot: np.ndarray) -> str:
        """
        Convert one-hot encoding to DNA sequence.

        Parameters
        ----------
        seq_1hot : np.ndarray, shape=(seq_len, 4)
            One-hot encoded sequence

        Returns
        -------
        seq : str
            DNA sequence
        """
        alphabet = ['A', 'C', 'G', 'T']
        seq_list = []

        for i in range(seq_1hot.shape[0]):
            if seq_1hot[i].sum() == 0:
                seq_list.append('N')
            else:
                seq_list.append(alphabet[seq_1hot[i].argmax()])

        return ''.join(seq_list)

    def _reverse_complement(self, seq: str) -> str:
        """
        Get reverse complement of DNA sequence.

        Parameters
        ----------
        seq : str
            DNA sequence

        Returns
        -------
        rc_seq : str
            Reverse complement
        """
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
                      'a': 't', 'c': 'g', 'g': 'c', 't': 'a', 'n': 'n'}
        return ''.join(complement.get(base, 'N') for base in reversed(seq))

    def __del__(self):
        """Close genome file when object is deleted."""
        if hasattr(self, 'genome_open'):
            self.genome_open.close()


class GeneSequence:
    """Container for a gene sequence with metadata."""

    def __init__(self, seq: str, seq_1hot: np.ndarray, gene_id: str,
                 abs_position: Tuple[int, int], rel_position: Tuple[int, int],
                 chr: str, strand: str, atg_pos: int, atg_found: bool,
                 cds_start: int, cds_end: int):
        """
        Initialize GeneSequence.

        Parameters
        ----------
        seq : str
            DNA sequence (ACGT)
        seq_1hot : np.ndarray, shape=(seq_len, 4)
            One-hot encoded sequence
        gene_id : str
            Gene identifier
        abs_position : tuple of (int, int)
            Absolute genomic coordinates (1-based, inclusive)
        rel_position : tuple of (int, int)
            Relative position to ATG (e.g., -16384 to +16383 for 32kb)
        chr : str
            Chromosome name
        strand : str
            '+' or '-'
        atg_pos : int
            ATG position (1-based genomic coordinate)
        atg_found : bool
            Whether ATG was found at expected center position
        cds_start : int
            CDS start position (1-based, -1 if not available)
        cds_end : int
            CDS end position (1-based, -1 if not available)
        """
        self.seq = seq
        self.seq_1hot = seq_1hot
        self.gene_id = gene_id
        self.abs_position = abs_position
        self.rel_position = rel_position
        self.chr = chr
        self.strand = strand
        self.atg_pos = atg_pos
        self.atg_found = atg_found
        self.cds_start = cds_start
        self.cds_end = cds_end

    def __repr__(self):
        return (f"GeneSequence(gene_id='{self.gene_id}', chr={self.chr}, "
                f"strand={self.strand}, length={len(self.seq)}, "
                f"atg_found={self.atg_found})")
