import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.ticker import FuncFormatter

def extract_layer_from_trace(trace):
    """
    Extracts the layer information from a trace path by taking everything
    before the first occurrence of '/Traces/' (case-insensitive).
    If '/Traces/' is not found, it falls back to returning the portion
    up to (and including) the first folder that starts with 'layer_'.
    """
    m = re.search(r'(?i)(.*)/traces/', trace)
    if m:
        return m.group(1)
    else:
        parts = trace.split('/')
        for i, part in enumerate(parts):
            if part.startswith("layer_"):
                return '/'.join(parts[:i+1])
        return trace

def load_and_prepare_data(csv_file, layer=None, per_layer=False, normalize=False):
    """
    Loads the CSV file into a DataFrame, optionally filters by 'layer',
    and optionally normalizes.

    NOTE: We do NOT do any log transform here, because we want to:
      1) Always compute Pearson correlation on the original (or normalized) data.
      2) If the user wants to *visualize* logs, we will handle that in the plotting step.

    Returns:
        pd.DataFrame (potentially filtered and normalized)
    """
    df = pd.read_csv(csv_file)
    
    # If no 'Layer' column but a 'Trace' column exists, extract layer info
    if 'Layer' not in df.columns and 'Trace' in df.columns:
        df['Layer'] = df['Trace'].apply(extract_layer_from_trace)
    
    # If a specific layer is provided (and we have a 'Layer' column), filter
    if layer is not None and 'Layer' in df.columns:
        df = df[df['Layer'] == layer].copy()
    
    # If normalize is True, each column is divided by its mean
    # (We assume 'Analytical' and 'Simulation' exist and are numeric)
    if normalize:
        df['Analytical'] = df['Analytical'] / df['Analytical'].mean()
        df['Simulation'] = df['Simulation'] / df['Simulation'].mean()
    
    return df

def compute_and_plot_correlation(df, ax, title="", scatter_kws=None, line_kws=None, log_transform=False):
    """
    1) Compute Pearson correlation on the data in df['Analytical'], df['Simulation'].
    2) Create a *copy* for plotting (df_plot).
    3) If log_transform=True, filter out <= 0 and do log10 transform just for plotting.
    4) Plot a regression line with seaborn's regplot on df_plot.
    5) Apply a custom ticker if log_transform is True.
    6) Show R^2 instead of R in the text annotation.
    """
    if scatter_kws is None:
        scatter_kws = dict(s=50)
    if line_kws is None:
        line_kws = dict(color='red')
    
    # --- 1) Pearson correlation on the "df" data (original or normalized) ---
    corr_coef = df['Analytical'].corr(df['Simulation'])
    r_value, p_value = pearsonr(df['Analytical'], df['Simulation'])
    
    # --- 2) Make a copy for plotting ---
    df_plot = df.copy()
    
    # --- 3) If log_transform, filter out <= 0 and do log10 transform just for plotting
    if log_transform:
        df_plot = df_plot[(df_plot['Analytical'] > 0) & (df_plot['Simulation'] > 0)].copy()
        df_plot['Analytical'] = np.log10(df_plot['Analytical'])
        df_plot['Simulation'] = np.log10(df_plot['Simulation'])
    
    # --- 4) regplot on df_plot ---
    sns.regplot(
        x='Simulation',
        y='Analytical',
        data=df_plot,
        ci=95,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        ax=ax
    )
    
    # Calculate R^2 from r_value
    r_squared = r_value**2
    
    # Place correlation annotation (now using R^2)
    ax.text(
        0.05,
        0.95,
        rf"$R^2 = {r_squared:.3f}$",
        transform=ax.transAxes,
        fontsize=16,  # Larger font size for the annotation
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )
    
    # Place the title at the bottom of the subplot
    # t = ax.set_title(title, fontsize=16, fontweight='bold')
    # t.set_position([0.5, -0.2])  # (x=0.5 means center, y < 0 means below the x-axis)
    
    ax.text(
        0.5, -0.32,  # Move down as needed (adjust -0.25 => -0.15, etc.)
        title,
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=16,
        fontweight='bold'
    )
    
    ax.set_xlabel("Simulation (Cycles)", fontsize=16)
    ax.set_ylabel("Analytical (Cycles)", fontsize=16)
    
    # Increase tick-label font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # 5) If log_transform, apply custom tick formatter to show 10^x
    if log_transform:
        log_formatter = FuncFormatter(lambda x, pos: rf"$10^{{{x:.0f}}}$")
        ax.xaxis.set_major_formatter(log_formatter)
        ax.yaxis.set_major_formatter(log_formatter)
    
    # 6) Add more ticks on the y-axis
    ax.locator_params(axis='y', nbins=8)
    
    ax.grid(True)
    return corr_coef, r_value, p_value

def analyze_two_csvs(
    csv_file1,
    csv_file2,
    output_file=None,
    layer=None,
    per_layer=False,
    normalize=False,
    log_transform=False
):
    """
    1) Loads two CSV files into DataFrames (applying layer filtering and/or normalization).
    2) Creates a figure with two subplots (stacked vertically).
    3) In each subplot:
       - We compute correlation on the DataFrame's original or normalized columns.
       - If log_transform=True, we *only* transform the plot, not the correlation.
    """
    
    # 1. Load and prepare data for CSV1 (correlation data in df1)
    df1 = load_and_prepare_data(
        csv_file1,
        layer=layer,
        per_layer=per_layer,
        normalize=normalize
    )
    
    # 2. Load and prepare data for CSV2 (correlation data in df2)
    df2 = load_and_prepare_data(
        csv_file2,
        layer=layer,
        per_layer=per_layer,
        normalize=normalize
    )
    
    # 3. Create figure with 2 subplots, new size is (10, 6)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    
    # 4. Plot correlation for CSV1
    title1 = "(a) Correlation: Analytical vs. Simulation (HBM-PIM)"
    compute_and_plot_correlation(
        df1,
        ax=ax1,
        title=title1,
        log_transform=log_transform
    )
    
    # 5. Plot correlation for CSV2
    title2 = "(b) Correlation: Analytical vs. Simulation (SIMDRAM)"
    compute_and_plot_correlation(
        df2,
        ax=ax2,
        title=title2,
        log_transform=log_transform
    )
    
    # Adjust layout
    plt.tight_layout()
    # Optionally expand bottom margin if the bottom titles are cut off:
    # plt.subplots_adjust(bottom=0.15, hspace=0.3)
    
    # 6. Save or show
    if output_file:
        plt.savefig(output_file, format='pdf', bbox_inches='tight')
        print(f"Combined figure saved to {output_file}")
    else:
        plt.show()
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze correlation for two CSV files, optionally with layer filtering, normalization, and log transform on the axes (not the correlation).'
    )
    
    # Two CSV inputs
    parser.add_argument('--data_file1', type=str, required=True, help='Path to the first CSV file.')
    parser.add_argument('--data_file2', type=str, required=True, help='Path to the second CSV file.')
    
    # Output PDF
    parser.add_argument('--output_file', type=str, help='Path to save the combined figure as PDF.')
    
    # Optional flags
    parser.add_argument('--normalize', action='store_true', help='Normalize data by dividing by mean (for correlation and plotting).')
    parser.add_argument('--log_transform', action='store_true', 
                        help='If set, display the scatterplot axes in log10 scale (but correlation is computed on the original data).')
    
    # Layer filtering
    parser.add_argument('--layer', type=str, default=None,
                        help="Analyze only the specified layer (requires 'Layer' column or 'Trace' column).")
    parser.add_argument('--per_layer', action='store_true',
                        help='Perform analysis for each unique layer separately (not demonstrated in detail here).')
    
    args = parser.parse_args()
    
    analyze_two_csvs(
        csv_file1=args.data_file1,
        csv_file2=args.data_file2,
        output_file=args.output_file,
        layer=args.layer,
        per_layer=args.per_layer,
        normalize=args.normalize,
        log_transform=args.log_transform
    )
