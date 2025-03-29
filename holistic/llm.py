from __future__ import annotations

from calCycle import calculate_total_cycles
from typing import List
from database import Database, MAX_VAL
import json
import os
import logging
from logging import Logger
from tqdm import tqdm

RESULTS_PATH = "exp_results/fig15/"

DATABASE_FILE = os.path.join(RESULTS_PATH, "all_layer_llama_ae.json")

MODEL_FILE_PATH = "nn_models/"

# FIXME: change models accordingly, maybe no need to run 8B and 14B separately
MODELS: List[str] = ["llama3-8B-128", "llama3-8B-256", "llama3-8B-512", "llama3-8B-1024", "llama3-14B-128", "llama3-14B-256", "llama3-14B-512", "llama3-14B-1024"]

def get_cycle(model_name: str, arch: str, n_banks: int, grouping: bool, recurrent: bool, logger: Logger) -> float:
    model_file = os.path.join(MODEL_FILE_PATH, f"{model_name}/layer_params.csv")
    database = Database(DATABASE_FILE, model_name, arch, logger)
    total_cycles = calculate_total_cycles(model_file, n_banks, database, grouping, recurrent, logger)

    return total_cycles

def main(logger: Logger) -> None:
    results = []
    # Adjust recurrent as needed; here we assume 'False'
    recurrent = True
    # FIXME: change n_banks to the setting
    n_banks = 8192
    grouping = True
    arch = "hbm"
    
    # Use product() to combine all parameters into one iterable
    for model in tqdm(MODELS):
        cycles: float = get_cycle(model, arch, n_banks, grouping, recurrent, logger)
        if cycles == MAX_VAL:
            cycles = -1
        results.append({
            "model": model,
            "arch": arch,
            "n_banks": n_banks,
            "grouping": grouping,
            "cycles": cycles,
        })
        # logger.info(f"Model: {model}, Arch: {arch}, #Banks: {n_banks}, Grouping: {grouping}, Cycles: {cycles}")
    
    # Dump results to JSON
    output_file = os.path.join(RESULTS_PATH, "holistic_llama.json")
    with open(output_file, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("llm")
    main(logger)