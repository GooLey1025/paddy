# Harrow: TensorFlow Interpretation Toolkit for Rice Models

**Harrow** is a comprehensive TensorFlow/Keras-based interpretation library for rice regulatory models, specifically designed to work with Paddy's SeqNN architecture. It is inspired by and migrated from [tangermeme](https://github.com/jmschrei/tangermeme), adapting PyTorch-based genomic interpretation methods to TensorFlow.

## Overview

Harrow provides a complete suite of tools for interpreting deep learning models in genomics:

- **Attribution Methods**: Integrated Gradients, vanilla gradients, in-silico saturation mutagenesis
- **Perturbation Experiments**: Marginalization, ablation, variant effect prediction
- **Sequence Manipulation**: Fast one-hot encoding, shuffling, substitution, dinucleotide shuffling
- **Advanced Analysis**: Motif spacing, cartesian product analysis, sequence design
- **Data I/O**: FASTA/BED file loading, sequence extraction
- **Visualization**: Attribution plots, heatmaps, sequence logos

## Installation

Harrow is part of the Paddy package:

```bash
cd /home/gulei/projects/ricereg/paddy
pip install -e .
```

## Quick Start

```python
import harrow
import numpy as np

# One-hot encode a DNA sequence (fast numba-based)
seq_onehot = harrow.one_hot_encode("ACGTACGT")  # Shape: (8, 4)

# Generate random sequences
X = harrow.random_one_hot((100, 2000, 4))  # 100 sequences of length 2000

# Load a Paddy model and make predictions
from paddy.seqnn import SeqNN
import json

with open('model_params.json') as f:
    params = json.load(f)
model = SeqNN(params)
model.restore('model_weights.h5')

# Make predictions
predictions = harrow.predict(model, X, batch_size=32)

# Compute integrated gradients
attributions = harrow.integrated_gradients(
    model, X, 
    baseline=None,  # Use zero baseline
    num_steps=50,
    batch_size=16
)

# Perform marginalization experiment
from harrow import marginalize, substitute
X_background = harrow.random_one_hot((50, 2000, 4))
motif = "TATAAA"  # TATA box

y_before, y_after = marginalize(model, X_background, motif)
effect = (y_after - y_before).mean(axis=0)  # Average effect across backgrounds

# In-silico saturation mutagenesis
ism_scores = harrow.saturation_mutagenesis(
    model, X[0:1], 
    start=900, end=1100,  # Focus on central 200bp
    batch_size=32
)

# Visualize
import matplotlib.pyplot as plt
harrow.plot_attribution(attributions[0], figsize=(20, 3))
plt.show()
```

## Key Features

### Fast One-Hot Encoding

Harrow uses numba-optimized one-hot encoding (from tangermeme), which is significantly faster than standard implementations:

```python
# Fast encoding with numba
seq_onehot = harrow.one_hot_encode("ACGTACGT", alphabet=['A','C','G','T'])

# Decode back to string
seq_str = harrow.characters(seq_onehot)
```

### Sequence Manipulation (Ersatz)

```python
from harrow import substitute, shuffle, dinucleotide_shuffle

# Substitute a motif into sequences
X_modified = substitute(X, motif="GATA", start=500)

# Shuffle a region (destroys signal)
X_shuffled = shuffle(X, start=400, end=600, random_state=42)

# Dinucleotide shuffle (preserves dinucleotide frequencies)
X_dinuc = dinucleotide_shuffle(X, start=400, end=600, random_state=42)
```

### Attribution Methods

```python
# Integrated Gradients (recommended)
ig_attr = harrow.integrated_gradients(model, X, num_steps=50)

# Vanilla gradients
grad_attr = harrow.gradients(model, X)

# In-silico saturation mutagenesis
ism_attr = harrow.saturation_mutagenesis(model, X, start=0, end=-1)
```

### Perturbation Experiments

```python
# Marginalization: test motif effect on background
y_before, y_after = harrow.marginalize(model, X_background, motif="CACGTG")

# Ablation: shuffle region and measure effect
y_original, y_shuffled_list = harrow.ablate(
    model, X, start=500, end=700, n=20, random_state=42
)

# Variant effect: test specific variants
substitutions = np.array([
    [0, 1000, 2],  # Example 0, position 1000, nucleotide G (index 2)
    [1, 1500, 1],  # Example 1, position 1500, nucleotide C (index 1)
])
y_ref, y_alt = harrow.substitution_effect(model, X, substitutions)
```

### Visualization

```python
import harrow

# Attribution plot (sequence logo style)
harrow.plot_attribution(attributions[0], figsize=(20, 3))

# ISM heatmap
harrow.plot_ism(ism_scores[0], figsize=(20, 4))

# Marginalization heatmap
import pandas as pd
tissue_names = pd.read_csv('tissue_names.txt', header=None)[0].values
y_delta = y_after - y_before
harrow.plot_marginalization_heatmap(
    y_delta.T, 
    names=tissue_names,
    figsize=(10, 8)
)
```

## Architecture

Harrow is designed to work seamlessly with Paddy's SeqNN models:

- **Input format**: (batch, seq_length, 4) one-hot encoded sequences
- **Model compatibility**: Supports SeqNN's `model_type='2d_to_1d'` architecture
- **Multi-head support**: Works with models that have multiple output heads
- **Parameter passing**: Supports SeqNN-specific parameters like `track_scale`, `track_transform`

## Module Overview

- **utils.py**: Core utilities (one-hot encoding, validation)
- **ersatz.py**: Sequence manipulation operations
- **predict.py**: Batched prediction wrapper
- **attribution.py**: Attribution methods (IG, gradients)
- **ism.py**: In-silico saturation mutagenesis
- **marginalize.py**: Marginalization experiments
- **ablate.py**: Ablation experiments
- **variant.py**: Variant effect prediction
- **space.py**: Motif spacing analysis
- **product.py**: Cartesian product analysis
- **design.py**: Sequence design/optimization
- **io.py**: Data loading utilities
- **seqlet.py**: Seqlet extraction
- **annotate.py**: Annotation-based analysis
- **match.py**: Motif matching/scanning
- **kmers.py**: K-mer analysis
- **pisa.py**: Positional importance scoring
- **plot.py**: Visualization functions

## Tutorials

Comprehensive Jupyter notebook tutorials are available in `/home/gulei/projects/ricereg/paddy/docs/`:

### Group A: Core Functionality
- Tutorial_A1_Ersatz_Sequence_Manipulation.ipynb
- Tutorial_A2_Predictions.ipynb
- Tutorial_A3_Attribution_Methods.ipynb
- Tutorial_A4_Seqlets.ipynb
- Tutorial_A5_Annotations.ipynb

### Group B: Perturbation Experiments
- Tutorial_B1_Marginalization.ipynb
- Tutorial_B2_Ablation.ipynb
- Tutorial_B3_Spacing.ipynb
- Tutorial_B4_Saturation_Mutagenesis.ipynb
- Tutorial_B5_Variant_Effect.ipynb
- Tutorial_B6_Design.ipynb
- Tutorial_B7_Cartesian_Product.ipynb

### Group C: Data and Visualization
- Tutorial_C1_IO_and_Data_Loading.ipynb
- Tutorial_C2_Plotting.ipynb

## Differences from Tangermeme

While harrow is inspired by tangermeme, there are key differences:

1. **Framework**: TensorFlow/Keras instead of PyTorch
2. **Model integration**: Designed for Paddy's SeqNN architecture
3. **Attribution backend**: Wraps existing TensorFlow GradientTape-based methods in SeqNN
4. **Data format**: Uses (batch, length, channels) instead of (batch, channels, length)
5. **Rice genomics focus**: Examples and tutorials use rice regulatory sequences

