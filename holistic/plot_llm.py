import numpy as np
import matplotlib.pyplot as plt

import json
import os

RESULT_PATH = "exp_results/fig15/"

def parse_llama_latency(json_path: str):
    """
    Parses a list of dictionaries containing Llama model cycle data
    and returns a nested list in the format:
    
        [
            [<8B@128_tokens>, <14B@128_tokens>],
            [<8B@256_tokens>, <14B@256_tokens>],
            [<8B@512_tokens>, <14B@512_tokens>],
            [<8B@1024_tokens>, <14B@1024_tokens>]
        ]
    
    where each value is the latency in seconds at 1GHz.
    
    Parameters
    ----------
    json_data : str or list
        If a string, it should be valid JSON. If a list, it should be
        the loaded JSON (i.e., a list of dicts).
    
    Returns
    -------
    list
        Nested list of floats representing latencies in the order:
        [[8B_128, 14B_128],
         [8B_256, 14B_256],
         [8B_512, 14B_512],
         [8B_1024, 14B_1024]]
    """

    # If json_data is a string, parse it into a Python object
    with open(json_path, 'r') as f:
        json_file = json.load(f)

    # We'll store the latencies in a dict keyed by token_length, holding a sub-dict for 8B and 14B
    # E.g. latencies["128"] = {"8B": 0.13, "14B": 0.24}
    latencies = {}

    for item in json_file:
        # "model" might look like: "llama3-8B-128"
        # We'll split by '-' to extract the size and token-length
        # model_split = ["llama3", "8B", "128"]
        model_split = item["model"].split('-')
        size = model_split[1]   # e.g. "8B"
        tokens = model_split[2] # e.g. "128"

        # Convert cycles to seconds at 1GHz
        latency_seconds = item['cycles'] / 1e9

        if tokens not in latencies:
            latencies[tokens] = {}
        latencies[tokens][size] = latency_seconds

    # Now we assemble the final nested list in ascending order of token length
    # for the sizes: 8B then 14B
    ordered_tokens = ["128", "256", "512", "1024"]
    result = []
    for tok in ordered_tokens:
        # Each entry is [8B_latency, 14B_latency] for the given token length
        result.append([
            latencies[tok]["8B"],
            latencies[tok]["14B"]
        ])

    return result


def plot_latency(model_latency_diff_token_length):
    # Fixed data for token lengths and model names
    token_length = ["128", "256", "512", "1024"]
    model_name = ["Llama-3 8B", "Llama-3 14B"]

    n_groups = len(token_length)
    n_models = len(model_name)

    # X positions for each group
    x = np.arange(n_groups)
    bar_width = 0.35  # width of each bar

    fig, ax = plt.subplots(figsize=(6, 2))

    # Plot each model as a separate set of bars
    for i, model in enumerate(model_name):
        y_values = [row[i] for row in model_latency_diff_token_length]

        # Calculate the bar positions for this model
        bar_positions = x + i * bar_width
        
        ax.bar(
            bar_positions,
            y_values,
            bar_width,
            color=plt.cm.tab20c(4 * i + i), # type: ignore
            edgecolor='black',
            label=model
        )

    # Center the tick labels between the grouped bars
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(token_length, fontsize=10)

    # Draw grid lines on the y-axis
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7)

    # Labels
    ax.set_xlabel("Sequence Length (batch = 1)", fontsize=12)
    ax.set_ylabel("Latency @1GHz (s)", fontsize=12)

    ax.legend(fontsize=10)

    # Save to PDF (adjust the path if needed)
    fig_path = os.path.join(RESULT_PATH, "fig15.pdf")
    plt.savefig(fig_path, bbox_inches='tight', format="pdf")
    plt.close(fig)

if __name__ == "__main__":
    json_path = os.path.join(RESULT_PATH, "holistic_llama.json")
    data = parse_llama_latency(json_path)
    plot_latency(data)