import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_stiefel_analysis_separated(csv_path="/ssd1/zhizhou/workspace/rotation-project/replace/plot_reproduce/stiefel_analysis_metrics.csv", output_dir="/ssd1/zhizhou/workspace/rotation-project/replace/plot_reproduce/plots"):
    # 1. Setup
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 2. Parse Module Names
    def parse_module(s):
        parts = s.split('.')
        prefix = parts[0] # e.g. "Lang_L10"
        domain, layer_str = prefix.split('_')
        layer_idx = int(layer_str.replace('L', ''))
        module_type = ".".join(parts[1:]) # e.g. "self_attn.q_proj"
        return pd.Series([domain, layer_idx, module_type])

    df[['Domain', 'Layer', 'ModuleType']] = df['Module'].apply(parse_module)
    
    # 3. Define Metrics to Plot Separately
    metrics_map = {
        "Spectrum_RelErr": "Spectrum Stability (Energy Change)",
        "Baseline_RelErr": "Baseline Drift (Actual Weight Change)",
        "Inner_RelErr": "Inner Stiefel Error (Rotation Only)",
        "Ambient_RelErr": "Ambient Stiefel Error (Transport/Drift)",
        "Baseline_MSE": "Baseline Drift MSE",
        "Inner_MSE": "Inner Stiefel MSE",
        "Ambient_MSE": "Ambient Stiefel MSE",
        "Baseline_MAE": "Baseline Drift MAE",
        "Inner_MAE": "Inner Stiefel MAE",
        "Ambient_MAE": "Ambient Stiefel MAE",
    }
    
    # 4. Generate Plots
    sns.set_theme(style="whitegrid")
    
    for domain in ["Lang", "Vis"]:
        domain_df = df[df['Domain'] == domain]
        if domain_df.empty: continue
            
        print(f"\nGenerating plots for Domain: {domain}...")
        
        for metric_col, metric_name in metrics_map.items():
            
            # Create a dedicated figure for this Metric
            # Row=1, Col=Transitions
            g = sns.relplot(
                data=domain_df,
                x="Layer", 
                y=metric_col,
                hue="ModuleType", 
                col="Transition",    # Side-by-side comparison of stages
                kind="line", 
                marker="o",
                height=4, 
                aspect=1.2,
                linewidth=2.5,
                palette="tab10",     # Distinct colors
                facet_kws={'sharey': False, 'sharex': True} # Let Y-axis scale adapt
            )
            
            # Formatting
            g.set(yscale="log") # Log scale is crucial for 1e-12 vs 1e-4
            if "RelErr" in metric_col:
                g.set_axis_labels("Layer Index", "Relative Error (Log Scale)")
            else:
                g.set_axis_labels("Layer Index", "Mean Squared Error (Log Scale)")        
            # Title
            g.fig.suptitle(f"[{domain}] {metric_name}", y=1.05, fontsize=16, fontweight='bold')
            
            # Add gridlines
            for ax in g.axes.flat:
                ax.grid(True, which="both", ls="-", alpha=0.2)
                
                # Add 'Machine Precision' reference line for Ambient/Spectrum plots
                if "Ambient" in metric_col or "Spectrum" in metric_col:
                    ax.axhline(1e-7, color='red', linestyle='--', alpha=0.5, label='Float32 Precision')

            # Save
            filename = f"{domain}_{metric_col}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  -> Saved: {filepath}")
            plt.close()

if __name__ == "__main__":
    plot_stiefel_analysis_separated()