
import gzip
from intervaltree import IntervalTree
import numpy as np
import pybedtools


class GenomicInterval:
    def __init__(self, start, end, chrom=None, strand=None):
        self.start = start
        self.end = end
        self.chrom = chrom
        self.strand = strand

    def __eq__(self, other):
        return self.start == other.start

    def __lt__(self, other):
        return self.start < other.start

    def __cmp__(self, x):
        if self.start < x.start:
            return -1
        elif self.start > x.start:
            return 1
        else:
            return 0

    def __str__(self):
        if self.chrom is None:
            label = "[%d-%d]" % (self.start, self.end)
        else:
            label = "%s:%d-%d" % (self.chrom, self.start, self.end)
        return label


class Gene:
    """Class for managing genes in an isoform-agnostic way, taking
    the union of exons across isoforms."""

    def __init__(self, chrom, strand, kv, name=None):
        self.chrom = chrom
        self.strand = strand
        self.kv = kv
        self.name = name
        self.exons = IntervalTree()
        self.cds_intervals = IntervalTree()

    def add_exon(self, start, end):
        """BED 0-indexing assumed."""
        self.exons[start:end] = True
    
    def add_cds(self, start, end):
        """BED 0-indexing assumed."""
        self.cds_intervals[start:end] = True

    def get_exons(self):
        self.exons.merge_overlaps()
        return sorted(self.exons)

    def midpoint(self):
        positions = []
        for exon in self.get_exons():
            positions += range(exon.begin, exon.end)
        midp = int(np.mean(positions))
        return midp

    def span(self):
        exon_starts = [exon.begin for exon in self.exons]
        exon_ends = [exon.end for exon in self.exons]
        return min(exon_starts), max(exon_ends)
    
    def get_cds(self):
        """Get sorted CDS intervals."""
        # Backward compatibility: ensure cds_intervals exists
        if not hasattr(self, 'cds_intervals'):
            self.cds_intervals = IntervalTree()
        self.cds_intervals.merge_overlaps()
        # notice it is right open interval
        return sorted(self.cds_intervals)
    
    def cds_start(self):
        """Get the start position of the first CDS (ATG position for + strand) (0-indexed position).
        Falls back to gene start if no CDS information available."""
        cds_list = self.get_cds()
        if not cds_list:
            # If no CDS, fall back to gene start (first exon start)
            gene_start, _ = self.span()
            return gene_start
        return min(cds.begin for cds in cds_list)
    
    def cds_end(self):
        """Get the end position of the last CDS (ATG position for - strand) (0-indexed position)."""
        cds_list = self.get_cds()
        if not cds_list:
            _, gene_end = self.span()
            return gene_end
        # Because it is right open interval, So need to minus 1 to get the last CDS end position (which is still 0-indexed position)
        return max(cds.end for cds in cds_list) - 1

    def output_slice_old(self, seq_start, seq_len, model_stride, span=False):
        gene_slice = []

        if span:
            gene_start, gene_end = self.span()

            # clip left boundaries
            gene_seq_start = max(0, gene_start - seq_start)
            gene_seq_end = max(0, gene_end - seq_start)

            # requires >50% overlap
            slice_start = int(np.round(gene_seq_start / model_stride))
            slice_end = int(np.round(gene_seq_end / model_stride))

            # clip right boundaries
            slice_max = int(seq_len / model_stride)
            slice_start = min(slice_start, slice_max)
            slice_end = min(slice_end, slice_max)

            gene_slice = range(slice_start, slice_end)

        else:
            for exon in self.get_exons():
                # clip left boundaries
                exon_seq_start = max(0, exon.begin - seq_start)
                exon_seq_end = max(0, exon.end - seq_start)

                # requires >50% overlap
                slice_start = int(np.round(exon_seq_start / model_stride))
                slice_end = int(np.round(exon_seq_end / model_stride))

                # clip right boundaries
                slice_max = int(seq_len / model_stride)
                slice_start = min(slice_start, slice_max)
                slice_end = min(slice_end, slice_max)

                gene_slice.extend(range(slice_start, slice_end))

        return np.array(gene_slice)

    def output_slice(
        self,
        seq_start,
        seq_len,
        model_stride,
        span=False,
        majority_overlap=False,
        old_version=False,
    ):
        if old_version:
            return self.output_slice_old(seq_start, seq_len, model_stride, span=span)

        gene_slice = []

        def clip_boundaries(slice_start, slice_end):
            slice_max = int(seq_len / model_stride)
            slice_start = min(slice_start, slice_max)
            slice_end = min(slice_end, slice_max)
            slice_start = max(slice_start, 0)
            slice_end = max(slice_end, 0)
            return slice_start, slice_end

        if span:
            gene_start, gene_end = self.span()

            # clip left boundaries
            gene_seq_start = max(0, gene_start - seq_start)
            gene_seq_end = max(0, gene_end - seq_start)

            # requires >50% overlap
            slice_start = int(np.round(gene_seq_start / model_stride))
            slice_end = int(np.round(gene_seq_end / model_stride))

            # clip boundaries
            slice_start, slice_end = clip_boundaries(slice_start, slice_end)

            # add to gene slice
            if slice_start < slice_end:
                gene_slice = range(slice_start, slice_end)

        else:
            for exon in self.get_exons():
                # clip left boundaries
                exon_seq_start = max(0, exon.begin - seq_start)
                exon_seq_end = max(0, exon.end - seq_start)

                if majority_overlap:
                    # requires >50% overlap
                    slice_start = int(np.round(exon_seq_start / model_stride))
                    slice_end = int(np.round(exon_seq_end / model_stride))
                else:
                    # any overlap
                    slice_start = int(np.floor(exon_seq_start / model_stride))
                    slice_end = int(np.ceil(exon_seq_end / model_stride))

                # clip boundaries
                slice_start, slice_end = clip_boundaries(slice_start, slice_end)

                # add to gene slice
                if slice_start < slice_end:
                    gene_slice.extend(range(slice_start, slice_end))

        # collapse overlaps
        gene_slice = np.unique(gene_slice)

        return gene_slice


class Transcriptome:
    def __init__(self, gtf_file):
        self.genes = {}
        self.read_gtf(gtf_file)

    def read_gtf(self, gtf_file):
        if gtf_file[-3:] == ".gz":
            gtf_in = gzip.open(gtf_file, "rt")
        else:
            gtf_in = open(gtf_file)

        # ignore header
        line = gtf_in.readline()
        while line and line[0] == "#":
            line = gtf_in.readline()

        while line:
            a = line.split("\t")
            if len(a) >= 9 and a[2] in ["exon", "CDS"]:
                chrom = a[0]
                start = int(a[3])
                end = int(a[4])
                strand = a[6]
                kv = gtf_kv(a[8])
                
                # Try different ways to get gene_id (backward compatibility)
                gene_id = None
                if "gene_id" in kv:
                    gene_id = kv["gene_id"]
                elif "ID" in kv:
                    # For GFF3-style format, extract gene part from ID
                    id_val = kv["ID"]
                    if ":" in id_val:
                        # Handle cases like "LOC_Os08g39850.1:exon_3"
                        gene_id = id_val.split(":")[0]
                    else:
                        gene_id = id_val
                    
                    # Remove isoform suffix (e.g., .1, .2, .3) to get primary gene ID
                    if "." in gene_id:
                        gene_id = gene_id.split(".")[0]
                elif "Parent" in kv:
                    # Use Parent as gene_id if available
                    gene_id = kv["Parent"]
                else:
                    # Skip this exon if we can't determine gene_id
                    print(f"Warning: Could not determine gene_id for exon at {chrom}:{start}-{end}")
                    line = gtf_in.readline()
                    continue
                
                # Try different ways to get gene_name
                gene_name = None
                if "gene_name" in kv:
                    gene_name = kv["gene_name"]
                elif "Name" in kv:
                    gene_name = kv["Name"]
                elif "gene_symbol" in kv:
                    gene_name = kv["gene_symbol"]

                # initialize gene
                if gene_id not in self.genes:
                    self.genes[gene_id] = Gene(chrom, strand, kv, gene_name)

                # add exon or CDS
                if a[2] == "exon":
                    self.genes[gene_id].add_exon(start - 1, end)
                elif a[2] == "CDS":
                    self.genes[gene_id].add_cds(start - 1, end)

            line = gtf_in.readline()

        gtf_in.close()

    def bedtool_exon(self):
        # assemble sequence bedtool
        bed_lines = []
        for gene_id, gene in self.genes.items():
            for exon in gene.get_exons():
                exon_line = "%s %d %d %s . %s" % (
                    gene.chrom,
                    exon.begin,
                    exon.end,
                    gene_id,
                    gene.strand,
                )
                bed_lines.append(exon_line)
        genes_bedt = pybedtools.BedTool("\n".join(bed_lines), from_string=True)
        return genes_bedt

    def bedtool_span(self):
        # assemble sequence bedtool
        bed_lines = []
        for gene_id, gene in self.genes.items():
            gene_start, gene_end = gene.span()
            span_line = "%s %d %d %s . %s" % (
                gene.chrom,
                gene_start,
                gene_end,
                gene_id,
                gene.strand,
            )
            bed_lines.append(span_line)
        genes_bedt = pybedtools.BedTool("\n".join(bed_lines), from_string=True)
        return genes_bedt

    def write_bed_exon(self, bed_file):
        pass

    def write_bed_span(self, bed_file):
        pass


################################################################################
# Methods
################################################################################
def gtf_kv(s):
    """Convert the last gtf section of key/value pairs into a dict."""
    d = {}
    
    # Split by semicolon to get individual key-value pairs
    a = s.split(";")
    
    for key_val in a:
        key_val = key_val.strip()
        if not key_val:
            continue
            
        # Check if this uses = separator (like ID=value)
        eq_i = key_val.find("=")
        if eq_i != -1:
            # Split on first = only
            key = key_val[:eq_i].strip()
            val = key_val[eq_i + 1:].strip()
            
            # Remove quotes if present
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
                
        else:
            # Fall back to space-separated format (like gene_id "value")
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
