from __future__ import annotations

from calCycle import calculate_total_cycles
from typing import List
from database import Database, MAX_VAL
import json
import os
import logging
from logging import Logger
from itertools import product
from tqdm import tqdm

RESULTS_PATH = "exp_results/fig14/"

DATABASE_FILE = os.path.join(RESULTS_PATH, "all_layer_results_ae.json")

MODEL_FILE_PATH = "nn_models/"

MODELS: List[str] = ["alexnet", "resnet50", "resnet152", "unet", "vgg16", "bert"]

N_BANKS: List[int] = [256, 512, 1024]

ARCHS: List[str] = ["hbm", "simdram"]

GROUPING: List[bool] = [True, False]

def get_cycle(model_name: str, arch: str, n_banks: int, grouping: bool, recurrent: bool, logger: Logger) -> float:
    model_file = os.path.join(MODEL_FILE_PATH, f"{model_name}/layer_params.csv")
    database = Database(DATABASE_FILE, model_name, arch, logger)
    total_cycles = calculate_total_cycles(model_file, n_banks, database, grouping, recurrent, logger)

    return total_cycles

def main(logger: Logger) -> None:
    results = []
    # Adjust recurrent as needed; here we assume 'False'
    recurrent = False
    
    # Calculate the total number of iterations
    total_iterations = len(MODELS) * len(ARCHS) * len(N_BANKS) * len(GROUPING)
    
    # Use product() to combine all parameters into one iterable
    for (model, arch, n_banks, grouping) in tqdm(
        product(MODELS, ARCHS, N_BANKS, GROUPING),
        total=total_iterations,
        desc="Processing"  # Optional: text label
    ):
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
    output_file = os.path.join(RESULTS_PATH, "holistic_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("holistic")
    main(logger)