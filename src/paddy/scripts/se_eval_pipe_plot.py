#!/usr/bin/env python3
"""
Comprehensive visualization tool for model evaluation metrics.
This script creates various visualizations from metrics data in TSV format.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from matplotlib.colors import LinearSegmentedColormap

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize model evaluation metrics')
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help='Input TSV file with metrics')
    parser.add_argument('-o', '--output', type=str, default='metric_visualizations', 
                        help='Output directory')
    parser.add_argument('--show', action='store_true', 
                        help='Display plots (use in interactive environment)')
    parser.add_argument('--dpi', type=int, default=300, 
                        help='DPI for saved figures')
    parser.add_argument('--figsize', type=str, default='auto', 
                        help='Figure size in inches, format: widthxheight (e.g. 12x10) or "auto"')
    parser.add_argument('--no-boxplot', action='store_true', 
                        help='Skip boxplot generation')
    parser.add_argument('--no-violin', action='store_true', 
                        help='Skip violin plot generation')
    parser.add_argument('--no-heatmap', action='store_true', 
                        help='Skip heatmap generation')
    parser.add_argument('--no-scatter', action='store_true', 
                        help='Skip scatter plot generation')
    parser.add_argument('--no-barplot', action='store_true', 
                        help='Skip bar plot generation')
    parser.add_argument('--no-radar', action='store_true', 
                        help='Skip radar chart generation')
    parser.add_argument('--style', type=str, default='whitegrid', 
                        choices=['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'],
                        help='Seaborn plot style')
    parser.add_argument('--palette', type=str, default='tab10', 
                        help='Color palette for plots')
    parser.add_argument('--fontscale', type=float, default=1.1, 
                        help='Font scale factor')
    return parser.parse_args()

def extract_model_name(used_model):
    """Extract model name (before third underscore)"""
    parts = used_model.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])
    else:
        return used_model

def parse_figsize(figsize_str):
    """Parse figure size from string"""
    if figsize_str == 'auto':
        return None
    try:
        width, height = map(float, figsize_str.split('x'))
        return (width, height)
    except:
        print(f"Warning: Could not parse figsize '{figsize_str}', using default.")
        return None

def create_boxplots(df, args, metrics):
    """Create boxplots for each metric by model group"""
    print("Creating boxplots...")
    
    # Determine figure size
    figsize = parse_figsize(args.figsize) or (16, 14)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold', y=0.98)

    # Test Loss boxplot
    sns.boxplot(data=df, x='model_group', y='test_loss', ax=axes[0,0], palette=args.palette)
    axes[0,0].set_title('Test Loss Distribution by Model Group', fontsize=14)
    axes[0,0].set_xlabel('Model Group', fontsize=12)
    axes[0,0].set_ylabel('Test Loss', fontsize=12)
    axes[0,0].tick_params(axis='x', rotation=45)

    # Test PearsonR boxplot
    sns.boxplot(data=df, x='model_group', y='test_PearsonR', ax=axes[0,1], palette=args.palette)
    axes[0,1].set_title('Test PearsonR Distribution by Model Group', fontsize=14)
    axes[0,1].set_xlabel('Model Group', fontsize=12)
    axes[0,1].set_ylabel('Test PearsonR', fontsize=12)
    axes[0,1].tick_params(axis='x', rotation=45)

    # Test R2 boxplot
    sns.boxplot(data=df, x='model_group', y='test_R2', ax=axes[1,0], palette=args.palette)
    axes[1,0].set_title('Test R² Distribution by Model Group', fontsize=14)
    axes[1,0].set_xlabel('Model Group', fontsize=12)
    axes[1,0].set_ylabel('Test R²', fontsize=12)
    axes[1,0].tick_params(axis='x', rotation=45)

    # Scatter plot: PearsonR vs R2
    unique_models = df['model_group'].unique()
    # Fix for MatplotlibDeprecationWarning
    if hasattr(plt.cm, 'get_cmap'):
        cmap = plt.cm.get_cmap(args.palette, len(unique_models))
    else:
        cmap = plt.colormaps[args.palette]
    
    for i, model in enumerate(unique_models):
        model_data = df[df['model_group'] == model]
        axes[1,1].scatter(model_data['test_PearsonR'], model_data['test_R2'], 
                          c=[cmap(i)], alpha=0.7, s=60, label=model)
    
    # Create a more readable legend
    if len(unique_models) > 6:
        axes[1,1].legend(title='Model Group', loc='center left', 
                         bbox_to_anchor=(1, 0.5), fontsize=10)
    else:
        axes[1,1].legend(title='Model Group', loc='lower right', fontsize=10)
    
    axes[1,1].set_xlabel('Test PearsonR', fontsize=12)
    axes[1,1].set_ylabel('Test R²', fontsize=12)
    axes[1,1].set_title('PearsonR vs R² by Model Group', fontsize=14)
    
    plt.tight_layout()
    boxplot_path = os.path.join(args.output, 'boxplots.png')
    plt.savefig(boxplot_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved boxplots to: {boxplot_path}")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)

def create_violinplots(df, args, metrics):
    """Create violin plots for each metric by model group"""
    print("Creating violin plots...")
    
    # Determine figure size
    figsize = parse_figsize(args.figsize) or (20, 7)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Detailed Performance Distribution by Model Group', fontsize=18, fontweight='bold')
    
    # Test Loss violin plot
    sns.violinplot(data=df, x='model_group', y='test_loss', ax=axes[0], palette=args.palette)
    axes[0].set_title('Test Loss Distribution', fontsize=14)
    axes[0].set_xlabel('Model Group', fontsize=12)
    axes[0].set_ylabel('Test Loss', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Test PearsonR violin plot
    sns.violinplot(data=df, x='model_group', y='test_PearsonR', ax=axes[1], palette=args.palette)
    axes[1].set_title('Test PearsonR Distribution', fontsize=14)
    axes[1].set_xlabel('Model Group', fontsize=12)
    axes[1].set_ylabel('Test PearsonR', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Test R2 violin plot
    sns.violinplot(data=df, x='model_group', y='test_R2', ax=axes[2], palette=args.palette)
    axes[2].set_title('Test R² Distribution', fontsize=14)
    axes[2].set_xlabel('Model Group', fontsize=12)
    axes[2].set_ylabel('Test R²', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    violin_path = os.path.join(args.output, 'violinplots.png')
    plt.savefig(violin_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved violin plots to: {violin_path}")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)

def create_heatmaps(df, args, metrics):
    """Create heatmaps for each metric"""
    print("Creating heatmaps...")
    
    for metric in metrics:
        # Determine figure size based on data dimensions
        n_identifiers = df['identifier'].nunique()
        n_models = df['model_group'].nunique()
        width = max(10, n_models * 0.8)
        height = max(8, n_identifiers * 0.3)
        figsize = parse_figsize(args.figsize) or (width, height)
        
        plt.figure(figsize=figsize)
        pivot_data = df.pivot_table(values=metric, index='identifier', columns='model_group')
        
        # Sort rows and columns for better visualization
        if metric == 'test_loss':
            # For loss, lower is better
            row_order = pivot_data.mean(axis=1).sort_values(ascending=False).index
            col_order = pivot_data.mean(axis=0).sort_values(ascending=True).index
        else:
            # For PearsonR and R2, higher is better
            row_order = pivot_data.mean(axis=1).sort_values(ascending=True).index
            col_order = pivot_data.mean(axis=0).sort_values(ascending=False).index
        
        pivot_data = pivot_data.loc[row_order, col_order]
        
        # Create heatmap with appropriate colormap
        cmap = 'coolwarm_r' if metric == 'test_loss' else 'coolwarm'
        ax = sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap=cmap, 
                         linewidths=.5, annot_kws={"size": 8})
        
        plt.title(f'{metric} by Model Group and Identifier', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        heatmap_path = os.path.join(args.output, f'heatmap_{metric}.png')
        plt.savefig(heatmap_path, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved heatmap for {metric} to: {heatmap_path}")
        
        if args.show:
            plt.show()
        else:
            plt.close()

def create_barplots(df, args, metrics):
    """Create bar plots showing mean performance by model group"""
    print("Creating bar plots...")
    
    # Calculate mean metrics by model group
    mean_metrics = df.groupby('model_group')[metrics].mean().reset_index()
    
    # Determine figure size
    figsize = parse_figsize(args.figsize) or (15, 12)
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle('Average Performance by Model Group', fontsize=18, fontweight='bold')

    # Sort models by performance
    for i, metric in enumerate(metrics):
        # Sort by metric (ascending for loss, descending for others)
        if metric == 'test_loss':
            sorted_data = mean_metrics.sort_values(by=metric, ascending=True)
        else:
            sorted_data = mean_metrics.sort_values(by=metric, ascending=False)
        
        # Create bar plot
        sns.barplot(data=sorted_data, x='model_group', y=metric, ax=axes[i], palette=args.palette)
        
        # Add value labels on top of bars
        for j, v in enumerate(sorted_data[metric]):
            axes[i].text(j, v + 0.01, f"{v:.4f}", ha='center', fontsize=9)
        
        axes[i].set_title(f'Average {metric}', fontsize=14)
        axes[i].set_xlabel('Model Group', fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    barplot_path = os.path.join(args.output, 'barplots.png')
    plt.savefig(barplot_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved bar plots to: {barplot_path}")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)

def create_radar_chart(df, args, metrics):
    """Create radar chart comparing model groups across metrics"""
    print("Creating radar chart...")
    
    # Calculate mean metrics by model group
    mean_metrics = df.groupby('model_group')[metrics].mean()
    
    # Normalize metrics to 0-1 scale for radar chart
    normalized = pd.DataFrame(index=mean_metrics.index)
    
    for metric in metrics:
        if metric == 'test_loss':
            # For loss, lower is better, so invert the normalization
            min_val = mean_metrics[metric].min()
            max_val = mean_metrics[metric].max()
            normalized[metric] = 1 - ((mean_metrics[metric] - min_val) / (max_val - min_val))
        else:
            # For other metrics, higher is better
            min_val = mean_metrics[metric].min()
            max_val = mean_metrics[metric].max()
            normalized[metric] = (mean_metrics[metric] - min_val) / (max_val - min_val)
    
    # Set up radar chart
    labels = [m.replace('test_', '') for m in metrics]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Determine figure size
    figsize = parse_figsize(args.figsize) or (12, 10)
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot each model group
    for i, model in enumerate(normalized.index):
        values = normalized.loc[model].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set chart properties
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Clockwise
    
    # Set labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    
    plt.title('Model Performance Comparison (Normalized)', fontsize=16, y=1.1)
    
    radar_path = os.path.join(args.output, 'radar_chart.png')
    plt.tight_layout()
    plt.savefig(radar_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved radar chart to: {radar_path}")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)

def create_scatter_matrix(df, args, metrics):
    """Create scatter plot matrix showing relationships between metrics"""
    print("Creating scatter plot matrix...")
    
    # Determine figure size
    figsize = parse_figsize(args.figsize) or (12, 10)
    
    # Create pairplot
    g = sns.pairplot(df, vars=metrics, hue='model_group', palette=args.palette,
                    diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50})
    
    g.fig.suptitle('Relationships Between Metrics', fontsize=16, y=1.02)
    
    # Save the plot
    scatter_path = os.path.join(args.output, 'scatter_matrix.png')
    plt.savefig(scatter_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved scatter plot matrix to: {scatter_path}")
    
    if args.show:
        plt.show()
    else:
        plt.close()

def main():
    """Main function to process data and create visualizations"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Set plot style
    sns.set_style(args.style)
    sns.set_context("paper", font_scale=args.fontscale)
    
    # Read the data
    df = pd.read_csv(args.input, sep='\t')
    
    # Process the first column to extract model name
    df['model_group'] = df['used_model'].apply(extract_model_name)
    
    # Define metrics columns
    metrics = ['test_loss', 'test_PearsonR', 'test_R2']
    
    # Display the processed data
    print("Processed data preview:")
    print(df[['model_group', 'identifier'] + metrics].head(10))
    print(f"\nFound {df['model_group'].nunique()} model groups and {df['identifier'].nunique()} identifiers")
    
    # Create visualizations based on user preferences
    if not args.no_boxplot:
        create_boxplots(df, args, metrics)
    
    if not args.no_violin:
        create_violinplots(df, args, metrics)
    
    if not args.no_heatmap:
        create_heatmaps(df, args, metrics)
    
    if not args.no_barplot:
        create_barplots(df, args, metrics)
    
    if not args.no_radar:
        create_radar_chart(df, args, metrics)
    
    if not args.no_scatter:
        create_scatter_matrix(df, args, metrics)

    # Print summary statistics
    print("\nSummary Statistics by Model Group:")
    summary_stats = df.groupby('model_group')[metrics].agg(['mean', 'std', 'min', 'max'])
    print(summary_stats)
    
    # Save summary statistics - fix for multi-index DataFrame
    stats_path = os.path.join(args.output, 'summary_stats.csv')
    
    # Create a more readable format for the CSV
    # First, reset the index to make model_group a column
    stats_flat = summary_stats.reset_index()
    
    # Then create separate DataFrames for each metric and merge them
    final_stats = pd.DataFrame({'model_group': stats_flat['model_group']})
    
    for metric in metrics:
        for stat in ['mean', 'std', 'min', 'max']:
            col_name = f"{metric}_{stat}"
            final_stats[col_name] = stats_flat[(metric, stat)]
    
    # Save the flattened, more readable statistics
    final_stats.to_csv(stats_path, index=False)
    print(f"\nSummary statistics saved to: {stats_path}")

    # Save processed data
    processed_path = os.path.join(args.output, 'processed_metrics.tsv')
    df.to_csv(processed_path, sep='\t', index=False)
    print(f"\nProcessed data saved to: {processed_path}")
    
    # Create a README file with plot descriptions
    readme_path = os.path.join(args.output, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("# Model Evaluation Visualizations\n\n")
        f.write(f"Generated from: {args.input}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Files Description:\n\n")
        
        if not args.no_boxplot:
            f.write("- boxplots.png: Box plots showing distribution of metrics across model groups\n")
        
        if not args.no_violin:
            f.write("- violinplots.png: Violin plots showing detailed distribution of metrics\n")
        
        if not args.no_heatmap:
            for metric in metrics:
                f.write(f"- heatmap_{metric}.png: Heatmap showing {metric} for each model-identifier pair\n")
        
        if not args.no_barplot:
            f.write("- barplots.png: Bar plots showing average performance for each model group\n")
        
        if not args.no_radar:
            f.write("- radar_chart.png: Radar chart comparing normalized performance across metrics\n")
        
        if not args.no_scatter:
            f.write("- scatter_matrix.png: Scatter plot matrix showing relationships between metrics\n")
        
        f.write("- summary_stats.csv: Statistical summary of metrics by model group\n")
        f.write("- processed_metrics.tsv: Processed metrics data with model_group extracted\n")
    
    print(f"Created README file at: {readme_path}")
    print("\nVisualization complete!")

if __name__ == "__main__":
    main() 