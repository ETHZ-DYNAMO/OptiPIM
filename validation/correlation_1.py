import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

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

def analyze_analytical_simulation(csv_file, output_file=None, normalize=False, layer=None, per_layer=False):
    """
    Loads a CSV file, calculates the correlation between 'Analytical' and 'Simulation',
    and plots a scatter plot with a regression line (with Simulation on the x-axis and 
    Analytical on the y-axis). The Pearson r value is annotated near the regression line.
    Optionally, the values can be normalized by dividing by their mean.
    
    If a dedicated 'Layer' column is not present but a 'Trace' column exists, the function 
    extracts the layer information by taking everything before the first occurrence of '/Traces/'.
    
    You can either:
      - Analyze all data together.
      - Filter the analysis to a specific layer (using the `layer` parameter).
      - Perform analysis separately for each unique layer (using the `per_layer` flag).
    
    Parameters:
        csv_file (str): Path to the CSV file.
        output_file (str, optional): Path to save the plot as a PDF. If per_layer is True,
                                     the output_file name is used as a prefix for each layer.
        normalize (bool, optional): If True, each value in 'Analytical' and 'Simulation' is 
                                    divided by its mean. Defaults to False.
        layer (str, optional): If provided, only rows with layer equal to this value will be analyzed.
                               (Specify the full string—that is, everything before '/Traces/'.)
        per_layer (bool, optional): If True, perform analysis for each unique layer separately.
        
    Returns:
        If per_layer is False:
            tuple: (corr_coef, r_value, p_value) for the (filtered) data.
        If per_layer is True:
            dict: Mapping each layer to its (corr_coef, r_value, p_value).
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    print("First few rows of the data:")
    print(df.head())
    
    # If no 'Layer' column but a 'Trace' column exists, extract layer info.
    if 'Layer' not in df.columns and 'Trace' in df.columns:
        df['Layer'] = df['Trace'].apply(extract_layer_from_trace)
        print("Extracted 'Layer' column from 'Trace'. Unique layers found:")
        print(df['Layer'].unique())
    
    # Per-layer analysis: process each unique layer separately.
    if per_layer:
        if 'Layer' not in df.columns:
            print("No 'Layer' column found in the data. Cannot perform per-layer analysis.")
            return None
        
        results = {}
        for l in sorted(df['Layer'].unique()):
            print(f"\nProcessing layer: {l}")
            df_layer = df[df['Layer'] == l].copy()
            
            # Apply normalization if requested.
            if normalize:
                df_layer['Analytical_normalized'] = df_layer['Analytical'] / df_layer['Analytical'].mean()
                df_layer['Simulation_normalized'] = df_layer['Simulation'] / df_layer['Simulation'].mean()
                analytical_col = 'Analytical_normalized'
                simulation_col = 'Simulation_normalized'
                print(f"Layer {l}: Data normalized (values divided by their mean).")
            else:
                analytical_col = 'Analytical'
                simulation_col = 'Simulation'
            
            # Compute Pearson correlation.
            corr_coef = df_layer[analytical_col].corr(df_layer[simulation_col])
            r_value, p_value = pearsonr(df_layer[analytical_col], df_layer[simulation_col])
            print(f"Layer {l} - Pearson correlation (Pandas): {corr_coef:.3f}")
            print(f"Layer {l} - Pearson r (SciPy): {r_value:.3f} (p-value: {p_value:.3e})")
            
            # Create plot: Simulation on x-axis, Analytical on y-axis.
            plt.figure(figsize=(8, 6))
            sns.regplot(x=simulation_col, y=analytical_col, data=df_layer, ci=95,
                        scatter_kws={'s': 50}, line_kws={'color': 'red'})
            plt.title(f"Layer {l}: Correlation between Simulation (x) and Analytical (y)")
            plt.xlabel("Simulation" + (" (Normalized)" if normalize else ""))
            plt.ylabel("Analytical" + (" (Normalized)" if normalize else ""))
            plt.grid(True)
            
            # Annotate the plot with the r value near the regression line.
            plt.text(0.05, 0.95, f"R = {r_value:.3f}", transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # Save or display the plot.
            if output_file:
                # Sanitize the layer string for filenames (replace '/' with '_').
                safe_layer = l.replace('/', '_')
                base, ext = os.path.splitext(output_file)
                out_file = f"{base}_{safe_layer}{ext}"
                plt.savefig(out_file, format='pdf')
                print(f"Layer {l} plot saved as '{out_file}'.")
            else:
                plt.show()
            plt.clf()
            
            results[l] = (corr_coef, r_value, p_value)
        return results
    else:
        # If a specific layer is provided (and per_layer is False), filter the DataFrame.
        if layer is not None:
            if 'Layer' not in df.columns:
                print("No 'Layer' column found in the data; cannot filter by layer.")
            else:
                print(f"Filtering data for layer: {layer}")
                df = df[df['Layer'] == layer].copy()
        
        # Apply normalization if requested.
        if normalize:
            df['Analytical_normalized'] = df['Analytical'] / df['Analytical'].mean()
            df['Simulation_normalized'] = df['Simulation'] / df['Simulation'].mean()
            analytical_col = 'Analytical_normalized'
            simulation_col = 'Simulation_normalized'
            print("Data normalized (values divided by their mean).")
        else:
            analytical_col = 'Analytical'
            simulation_col = 'Simulation'
        
        # Compute Pearson correlation.
        corr_coef = df[analytical_col].corr(df[simulation_col])
        r_value, p_value = pearsonr(df[analytical_col], df[simulation_col])
        print(f"Pearson correlation (Pandas): {corr_coef:.3f}")
        print(f"Pearson r (SciPy): {r_value:.3f} (p-value: {p_value:.3e})")
        
        # Create the plot.
        plt.figure(figsize=(8, 6))
        sns.regplot(x=simulation_col, y=analytical_col, data=df, ci=95,
                    scatter_kws={'s': 50}, line_kws={'color': 'red'})
        plt.title("Correlation between Simulation (x) and Analytical (y)")
        plt.xlabel("Simulation" + (" (Normalized)" if normalize else ""))
        plt.ylabel("Analytical" + (" (Normalized)" if normalize else ""))
        plt.grid(True)
        
        # Annotate the plot with the r value near the regression line.
        plt.text(0.05, 0.95, f"R = {r_value:.3f}", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Save or display the plot.
        if output_file:
            plt.savefig(output_file, format='pdf')
            print(f"Plot saved as '{output_file}'.")
        else:
            plt.show()
        plt.clf()
        
        return corr_coef, r_value, p_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze correlation between Analytical and Simulation data with layer filtering via the Trace column.'
    )
    parser.add_argument('--data_file', type=str, required=True, help='Path to the CSV data file.')
    parser.add_argument('--output_file', type=str, help='Path to save the output PDF file.')
    parser.add_argument('--normalize', action='store_true', help='Normalize the data by dividing by its mean.')
    parser.add_argument('--layer', type=str, default=None,
                        help="Analyze only the specified layer (provide the full string before '/Traces/').")
    parser.add_argument('--per_layer', action='store_true',
                        help="Perform analysis for each unique layer separately (extracted from the 'Trace' column).")
    args = parser.parse_args()
    
    analyze_analytical_simulation(args.data_file, output_file=args.output_file,
                                  normalize=args.normalize, layer=args.layer, per_layer=args.per_layer)
