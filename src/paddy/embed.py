#!/usr/bin/env python

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from scipy import stats
import umap
import os
import warnings
warnings.filterwarnings('ignore')

"""
embed.py
Embedding analysis and comparison utilities for sequence models.
"""

def load_embeddings(file_path):
    """Load embeddings and metadata from HDF5 file."""
    print(f"Loading embeddings from: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Load embeddings
        embeddings = f['embeddings'][:]
        
        # Load metadata
        metadata = {}
        for attr_name in f.attrs:
            metadata[attr_name] = f.attrs[attr_name]
        
        # Load coordinates if available
        coordinates = None
        if 'coordinates' in f:
            chrom = [c.decode() for c in f['coordinates/chrom'][:]]
            start = f['coordinates/start'][:]
            end = f['coordinates/end'][:]
            coordinates = pd.DataFrame({
                'chrom': chrom,
                'start': start,
                'end': end
            })
        
        # Load sequences if available
        sequences = None
        if 'sequences' in f:
            sequences = [s.decode() for s in f['sequences'][:]]
    
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    
    return embeddings, metadata, coordinates, sequences

def analyze_embeddings(embeddings, metadata=None, verbose=True):
    """Basic analysis of embeddings."""
    if verbose:
        print("\n=== Embedding Analysis ===")
        print(f"Shape: {embeddings.shape}")
        print(f"Data type: {embeddings.dtype}")
        print(f"Memory usage: {embeddings.nbytes / 1024**2:.2f} MB")
    
    # Basic statistics
    stats_dict = {
        'shape': embeddings.shape,
        'min': embeddings.min(),
        'max': embeddings.max(),
        'mean': embeddings.mean(),
        'std': embeddings.std(),
        'nan_count': np.isnan(embeddings).sum(),
        'inf_count': np.isinf(embeddings).sum(),
        'memory_mb': embeddings.nbytes / 1024**2
    }
    
    if verbose:
        print(f"Min value: {stats_dict['min']:.4f}")
        print(f"Max value: {stats_dict['max']:.4f}")
        print(f"Mean: {stats_dict['mean']:.4f}")
        print(f"Std: {stats_dict['std']:.4f}")
        print(f"NaN count: {stats_dict['nan_count']}")
        print(f"Infinite count: {stats_dict['inf_count']}")
    
    # Add advanced statistics
    flat_embeddings = embeddings.flatten()
    stats_dict.update({
        'skewness': stats.skew(flat_embeddings),
        'kurtosis': stats.kurtosis(flat_embeddings),
        'q25': np.percentile(flat_embeddings, 25),
        'q75': np.percentile(flat_embeddings, 75),
    })
    
    if verbose:
        print(f"Skewness: {stats_dict['skewness']:.4f}")
        print(f"Kurtosis: {stats_dict['kurtosis']:.4f}")
        print(f"25th percentile: {stats_dict['q25']:.4f}")
        print(f"75th percentile: {stats_dict['q75']:.4f}")
        
        if metadata:
            print(f"\nSource layer: {metadata.get('layer_name', 'unknown')}")
            print(f"Layer type: {metadata.get('layer_type', 'unknown')}")
            print(f"Layer index: {metadata.get('layer_index', 'unknown')}")
    
    return stats_dict

def evaluate_embedding_quality(embeddings, verbose=True):
    """Evaluate embedding quality for pretrained model usage."""
    stats_dict = analyze_embeddings(embeddings, verbose=False)
    
    quality_scores = {}
    recommendations = []
    
    # 1. Distribution quality (ideal: mean~0, std~1, low skew/kurtosis)
    mean_score = max(0, 1 - abs(stats_dict['mean']) / 2)  # Penalty for deviation from 0
    std_score = max(0, 1 - abs(stats_dict['std'] - 1) / 2)  # Ideal std around 1
    skew_score = max(0, 1 - abs(stats_dict['skewness']) / 3)  # Low skewness
    kurt_score = max(0, 1 - abs(stats_dict['kurtosis']) / 5)  # Low kurtosis
    
    distribution_score = (mean_score + std_score + skew_score + kurt_score) / 4
    quality_scores['distribution'] = distribution_score
    
    # 2. Stability (no NaN/Inf values)
    stability_score = 1.0 if (stats_dict['nan_count'] == 0 and stats_dict['inf_count'] == 0) else 0.0
    quality_scores['stability'] = stability_score
    
    # 3. Dynamic range
    value_range = stats_dict['max'] - stats_dict['min']
    range_score = min(1.0, value_range / 10)  # Good if range > 10
    quality_scores['dynamic_range'] = range_score
    
    # Overall quality
    overall_quality = (distribution_score * 0.4 + stability_score * 0.4 + range_score * 0.2)
    quality_scores['overall'] = overall_quality
    
    # Generate recommendations
    if stats_dict['mean'] > 0.5 or stats_dict['mean'] < -0.5:
        recommendations.append("⚠️  Mean is far from 0 - consider layer normalization")
    
    if stats_dict['std'] < 0.1:
        recommendations.append("⚠️  Very low std - embeddings may lack diversity")
    elif stats_dict['std'] > 3:
        recommendations.append("⚠️  Very high std - embeddings may be unstable")
    
    if abs(stats_dict['skewness']) > 2:
        recommendations.append("⚠️  High skewness - distribution is asymmetric")
    
    if stats_dict['nan_count'] > 0 or stats_dict['inf_count'] > 0:
        recommendations.append("❌ NaN/Inf values detected - model may be unstable")
    
    if overall_quality > 0.8:
        recommendations.append("✅ Excellent embedding quality for pretrained use")
    elif overall_quality > 0.6:
        recommendations.append("✅ Good embedding quality for pretrained use")
    elif overall_quality > 0.4:
        recommendations.append("⚠️  Fair embedding quality - consider different layer")
    else:
        recommendations.append("❌ Poor embedding quality - not recommended for pretrained use")
    
    if verbose:
        print(f"\n=== Embedding Quality Assessment ===")
        print(f"Distribution Quality: {distribution_score:.3f}")
        print(f"Stability Score: {stability_score:.3f}")
        print(f"Dynamic Range Score: {range_score:.3f}")
        print(f"Overall Quality: {overall_quality:.3f}")
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  {rec}")
    
    return quality_scores, recommendations

def compare_embeddings(embedding_files, output_dir=None, verbose=True):
    """Compare multiple embedding files."""
    if verbose:
        print(f"\n=== Comparing {len(embedding_files)} Embedding Files ===")
    
    comparison_results = {}
    all_embeddings = {}
    all_metadata = {}
    
    # Load all embeddings
    for i, file_path in enumerate(embedding_files):
        embeddings, metadata, coordinates, sequences = load_embeddings(file_path)
        
        # Flatten embeddings for comparison
        if len(embeddings.shape) > 2:
            embeddings_flat = embeddings.mean(axis=1) if len(embeddings.shape) == 3 else embeddings.reshape(embeddings.shape[0], -1)
        else:
            embeddings_flat = embeddings
        
        file_key = f"file_{i}_{os.path.basename(file_path)}"
        all_embeddings[file_key] = embeddings_flat
        all_metadata[file_key] = metadata
        
        # Individual analysis
        stats_dict = analyze_embeddings(embeddings, metadata, verbose=False)
        quality_scores, recommendations = evaluate_embedding_quality(embeddings, verbose=False)
        
        comparison_results[file_key] = {
            'stats': stats_dict,
            'quality': quality_scores,
            'recommendations': recommendations,
            'file_path': file_path
        }
    
    # Cross-file comparison
    if len(embedding_files) > 1:
        if verbose:
            print(f"\n=== Cross-File Analysis ===")
        
        # Compare dimensions
        shapes = [all_embeddings[key].shape for key in all_embeddings.keys()]
        if len(set(shapes)) > 1:
            if verbose:
                print("⚠️  Warning: Embeddings have different shapes!")
                for key, shape in zip(all_embeddings.keys(), shapes):
                    print(f"  {key}: {shape}")
        
        # Correlation analysis between embeddings
        if all(shape == shapes[0] for shape in shapes):
            correlations = {}
            keys = list(all_embeddings.keys())
            
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    key1, key2 = keys[i], keys[j]
                    
                    # Compute correlation between flattened embeddings
                    flat1 = all_embeddings[key1].flatten()
                    flat2 = all_embeddings[key2].flatten()
                    
                    # Sample if too large for memory
                    if len(flat1) > 1000000:
                        indices = np.random.choice(len(flat1), 1000000, replace=False)
                        flat1 = flat1[indices]
                        flat2 = flat2[indices]
                    
                    corr = np.corrcoef(flat1, flat2)[0, 1]
                    correlations[f"{key1}_vs_{key2}"] = corr
                    
                    if verbose:
                        print(f"Correlation {key1} vs {key2}: {corr:.4f}")
        
        # Quality ranking
        quality_ranking = sorted(
            comparison_results.items(),
            key=lambda x: x[1]['quality']['overall'],
            reverse=True
        )
        
        if verbose:
            print(f"\n=== Quality Ranking ===")
            for rank, (key, results) in enumerate(quality_ranking, 1):
                quality = results['quality']['overall']
                layer_name = results.get('stats', {}).get('layer_name', 'unknown')
                print(f"{rank}. {key}: {quality:.3f} (layer: {layer_name})")
    
    # Save comparison report
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison plots
        plot_embedding_comparison(all_embeddings, all_metadata, output_dir)
        
        # Save detailed report
        save_comparison_report(comparison_results, output_dir)
    
    return comparison_results

def plot_embedding_distribution(embeddings, output_dir=None, title_suffix=""):
    """Plot distribution of embedding values."""
    plt.figure(figsize=(15, 5))
    
    # Flatten embeddings for histogram
    flat_embeddings = embeddings.flatten()
    
    plt.subplot(1, 4, 1)
    plt.hist(flat_embeddings, bins=50, alpha=0.7, density=True, color='skyblue')
    plt.xlabel('Embedding Value')
    plt.ylabel('Density')
    plt.title(f'Distribution{title_suffix}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 2)
    plt.boxplot(flat_embeddings, patch_artist=True, 
                boxprops=dict(facecolor='lightcoral', alpha=0.7))
    plt.ylabel('Embedding Value')
    plt.title(f'Boxplot{title_suffix}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 3)
    # Plot mean embedding per sequence
    if len(embeddings.shape) == 3:
        mean_per_seq = embeddings.mean(axis=(1, 2))
    else:
        mean_per_seq = embeddings.mean(axis=1)
    plt.hist(mean_per_seq, bins=30, alpha=0.7, density=True, color='lightgreen')
    plt.xlabel('Mean Embedding per Sequence')
    plt.ylabel('Density')
    plt.title(f'Sequence Means{title_suffix}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 4)
    # Q-Q plot for normality check
    stats.probplot(flat_embeddings, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot{title_suffix}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/embedding_distribution{title_suffix.replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def reduce_dimensions(embeddings, method='pca', n_components=2):
    """Reduce embedding dimensions for visualization."""
    print(f"Reducing dimensions using {method.upper()}...")
    
    # Flatten embeddings if they have more than 2 dimensions
    if len(embeddings.shape) > 2:
        if len(embeddings.shape) == 3:  # [n_sequences, seq_length, features]
            embeddings_flat = embeddings.mean(axis=1)  # Average across sequence length
        else:
            embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
    else:
        embeddings_flat = embeddings
    
    print(f"Input shape for dimensionality reduction: {embeddings_flat.shape}")
    
    # Check if we have enough samples for the requested number of components
    n_samples, n_features = embeddings_flat.shape
    max_components = min(n_samples, n_features)
    
    if n_components > max_components:
        print(f"Warning: Requested {n_components} components, but only {max_components} are possible.")
        print(f"Reducing n_components to {max_components}")
        n_components = max_components
    
    if n_samples < 2:
        print(f"Warning: Only {n_samples} sample(s) available. Cannot perform dimensionality reduction.")
        return embeddings_flat, None
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method.lower() == 'tsne':
        if n_samples < 2:
            print("t-SNE requires at least 2 samples")
            return embeddings_flat, None
        perplexity = min(30, (n_samples - 1) // 3, n_samples - 1)
        if perplexity < 1:
            perplexity = 1
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    elif method.lower() == 'umap':
        if n_samples < 2:
            print("UMAP requires at least 2 samples")
            return embeddings_flat, None
        reducer = umap.UMAP(n_components=n_components, random_state=42, 
                           n_neighbors=min(15, n_samples-1))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced = reducer.fit_transform(embeddings_flat)
    
    print(f"Reduced to shape: {reduced.shape}")
    return reduced, reducer

def plot_2d_embeddings(reduced_embeddings, coordinates=None, output_dir=None, method='pca', title_suffix=""):
    """Plot 2D reduced embeddings."""
    if reduced_embeddings is None or reduced_embeddings.shape[1] < 2:
        print(f"Cannot plot 2D embeddings: insufficient dimensions ({reduced_embeddings.shape if reduced_embeddings is not None else 'None'})")
        return
    
    plt.figure(figsize=(12, 8))
    
    if coordinates is not None and len(coordinates) == reduced_embeddings.shape[0]:
        # Color by chromosome if coordinates available
        unique_chroms = coordinates['chrom'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_chroms)))
        chrom_to_color = dict(zip(unique_chroms, colors))
        
        for chrom in unique_chroms:
            mask = coordinates['chrom'] == chrom
            if np.any(mask):
                plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                           c=[chrom_to_color[chrom]], label=chrom, alpha=0.7, s=50)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7, s=50)
    
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.title(f'2D Embedding Visualization ({method.upper()}){title_suffix}')
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(f"{output_dir}/embedding_2d_{method}{title_suffix.replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compute_similarity_matrix(embeddings, metric='cosine'):
    """Compute similarity matrix between sequences."""
    # Flatten embeddings if needed
    if len(embeddings.shape) > 2:
        embeddings_flat = embeddings.mean(axis=1) if len(embeddings.shape) == 3 else embeddings.reshape(embeddings.shape[0], -1)
    else:
        embeddings_flat = embeddings
    
    print(f"Computing {metric} similarity matrix...")
    
    if metric == 'cosine':
        similarity = cosine_similarity(embeddings_flat)
    elif metric == 'euclidean':
        distances = euclidean_distances(embeddings_flat)
        similarity = 1 / (1 + distances)  # Convert distance to similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return similarity

def cluster_embeddings(embeddings, n_clusters=None, method='kmeans'):
    """Cluster embeddings to assess separability."""
    # Flatten embeddings if needed
    if len(embeddings.shape) > 2:
        embeddings_flat = embeddings.mean(axis=1) if len(embeddings.shape) == 3 else embeddings.reshape(embeddings.shape[0], -1)
    else:
        embeddings_flat = embeddings
    
    if n_clusters is None:
        # Estimate optimal number of clusters using elbow method
        n_clusters = min(8, embeddings_flat.shape[0] // 2)
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(embeddings_flat)
        
        # Compute silhouette score
        from sklearn.metrics import silhouette_score
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(embeddings_flat, labels)
        else:
            silhouette_avg = 0
        
        return labels, silhouette_avg
    else:
        raise ValueError(f"Unknown clustering method: {method}")

def plot_similarity_matrix(similarity, output_dir=None, metric='cosine', title_suffix=""):
    """Plot similarity matrix heatmap."""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(similarity, cmap='viridis', square=True, 
                cbar_kws={'label': f'{metric.capitalize()} Similarity'})
    plt.title(f'Sequence Similarity Matrix ({metric.capitalize()}){title_suffix}')
    plt.xlabel('Sequence Index')
    plt.ylabel('Sequence Index')
    
    if output_dir:
        plt.savefig(f"{output_dir}/similarity_matrix_{metric}{title_suffix.replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_embedding_comparison(all_embeddings, all_metadata, output_dir):
    """Plot comparison between multiple embeddings."""
    n_files = len(all_embeddings)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Embedding Comparison', fontsize=16)
    
    # 1. Distribution comparison
    ax1 = axes[0, 0]
    for i, (key, embeddings) in enumerate(all_embeddings.items()):
        flat_emb = embeddings.flatten()
        ax1.hist(flat_emb, bins=50, alpha=0.6, density=True, 
                label=f"{key.split('_')[1]}")
    ax1.set_xlabel('Embedding Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Statistics comparison
    ax2 = axes[0, 1]
    stats_data = []
    labels = []
    for key, embeddings in all_embeddings.items():
        stats_data.append([embeddings.mean(), embeddings.std(), 
                          embeddings.min(), embeddings.max()])
        labels.append(key.split('_')[1])
    
    stats_df = pd.DataFrame(stats_data, columns=['Mean', 'Std', 'Min', 'Max'], index=labels)
    stats_df.plot(kind='bar', ax=ax2)
    ax2.set_title('Statistics Comparison')
    ax2.set_ylabel('Value')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. PCA comparison
    ax3 = axes[1, 0]
    has_comparison_data = False
    for i, (key, embeddings) in enumerate(all_embeddings.items()):
        if embeddings.shape[0] >= 2:  # Check if we have enough samples
            reduced, _ = reduce_dimensions(embeddings, method='pca', n_components=2)
            if reduced is not None and reduced.shape[1] >= 2:
                ax3.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7, 
                           label=f"{key.split('_')[1]}", s=30)
                has_comparison_data = True
    
    if has_comparison_data:
        ax3.set_xlabel('PCA 1')
        ax3.set_ylabel('PCA 2')
        ax3.set_title('PCA Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data\nfor PCA comparison\n(need ≥2 samples)', 
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('PCA Comparison')
    
    # 4. Quality scores comparison
    ax4 = axes[1, 1]
    # This would need quality scores - placeholder for now
    ax4.text(0.5, 0.5, 'Quality Scores\n(See detailed report)', 
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Quality Assessment')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/embedding_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_comparison_report(comparison_results, output_dir):
    """Save detailed comparison report."""
    report_path = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("Embedding Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        
        for key, results in comparison_results.items():
            f.write(f"File: {key}\n")
            f.write(f"Path: {results['file_path']}\n")
            f.write("-" * 30 + "\n")
            
            # Statistics
            stats = results['stats']
            f.write(f"Shape: {stats['shape']}\n")
            f.write(f"Mean: {stats['mean']:.4f}\n")
            f.write(f"Std: {stats['std']:.4f}\n")
            f.write(f"Min: {stats['min']:.4f}\n")
            f.write(f"Max: {stats['max']:.4f}\n")
            f.write(f"Skewness: {stats['skewness']:.4f}\n")
            f.write(f"Kurtosis: {stats['kurtosis']:.4f}\n")
            
            # Quality scores
            quality = results['quality']
            f.write(f"\nQuality Scores:\n")
            f.write(f"  Overall: {quality['overall']:.3f}\n")
            f.write(f"  Distribution: {quality['distribution']:.3f}\n")
            f.write(f"  Stability: {quality['stability']:.3f}\n")
            f.write(f"  Dynamic Range: {quality['dynamic_range']:.3f}\n")
            
            # Recommendations
            f.write(f"\nRecommendations:\n")
            for rec in results['recommendations']:
                f.write(f"  {rec}\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"Detailed comparison report saved to: {report_path}")
