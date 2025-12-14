# harrow: TensorFlow-based interpretation toolkit for rice regulatory models
# Migrated from tangermeme (PyTorch) to work with Paddy's SeqNN models
# Author: Adapted for TensorFlow/Keras

__version__ = '0.1.0'

# Core utilities
from .utils import (
    one_hot_encode,
    characters,
    random_one_hot,
    validate_input,
    reverse_complement,
    example_to_fasta_coords,
    extract_signal
)

# Sequence manipulation
from .ersatz import (
    substitute,
    insert,
    delete,
    shuffle,
    dinucleotide_shuffle,
    multisubstitute,
    randomize,
)

# Prediction
from .predict import predict

# Attribution methods
from .attribution import (
    integrated_gradients,
    gradients,
)

# In-silico saturation mutagenesis
from .ism import saturation_mutagenesis

# Perturbation experiments
from .marginalize import marginalize, marginalize_annotations
from .ablate import ablate, ablate_annotations
from .variant import (
    substitution_effect,
    deletion_effect,
    insertion_effect,
)

# Advanced analysis
from .space import space
from .product import cartesian_product
from .design import design

# Data I/O
from .io import extract_loci, read_bed, read_gff, read_fasta

# Seqlet analysis
from .seqlet import extract_seqlets, tfmodisco_seqlets, recursive_seqlets

# Annotation analysis
from .annotate import scan_sequences

# Motif matching
from .match import match_pwm

# K-mer analysis
from .kmers import kmer_frequencies

# Positional importance
from .pisa import positional_importance

# Visualization
from .plot import (
    plot_attribution,
    plot_weights,
    plot_marginalization_heatmap,
    plot_ism,
    plot_spacing_heatmap,
    plot_tracks,
    plot_logo,
    plot_categorical_scatter,
    plot_attributions,
    plot_pwm,
)

# Gene sequence extraction
from . import gene

__all__ = [
    # Core
    'one_hot_encode',
    'characters',
    'random_one_hot',
    'validate_input',
    'reverse_complement',
    'example_to_fasta_coords',
    'extract_signal',
    # Ersatz
    'substitute',
    'insert',
    'delete',
    'shuffle',
    'dinucleotide_shuffle',
    'multisubstitute',
    'randomize',
    # Prediction
    'predict',
    # Attribution
    'integrated_gradients',
    'gradients',
    'deep_lift_shap',
    'hypothetical_attributions',
    'saturation_mutagenesis',
    # Perturbation
    'marginalize',
    'marginalize_annotations',
    'ablate',
    'ablate_annotations',
    'substitution_effect',
    'deletion_effect',
    'insertion_effect',
    # Advanced
    'space',
    'cartesian_product',
    'design',
    # I/O
    'extract_loci',
    'read_bed',
    'read_gff',
    'read_fasta',
    # Seqlet
    'extract_seqlets',
    'tfmodisco_seqlets',
    'recursive_seqlets',
    # Annotate
    'scan_sequences',
    # Match
    'match_pwm',
    # Kmers
    'kmer_frequencies',
    # PISA
    'positional_importance',
    # Plot
    'plot_attribution',
    'plot_weights',
    'plot_marginalization_heatmap',
    'plot_ism',
    'plot_spacing_heatmap',
    'plot_tracks',
    'plot_logo',
    'plot_categorical_scatter',
    'plot_attributions',
    'plot_pwm',
    # Gene
    'gene',
]
