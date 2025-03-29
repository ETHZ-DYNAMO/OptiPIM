from __future__ import annotations
import json
import os
from typing import Dict
import re
from logging import Logger

MAX_VAL = int(1e20)

def mod_name_id(s: str, divisor: int) -> str:
    # Use a regular expression to extract the alphabet and number parts.
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", s)
    if not match:
        raise ValueError("Input must be in the format 'xxoo' with alphabets followed by digits.")
    
    alpha_part, number_part = match.groups()
    mod_value = int(number_part) % divisor
    return f"{alpha_part}{mod_value}"

class Database:
    def __init__(self, results_file: str, model: str, arch: str, logger: Logger) -> None:
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file {results_file} does not exist")
        
        logger.debug(f"Loading results from {results_file}...")
        with open(results_file) as f:
            file_data = json.load(f)
        
        logger.debug(f"Running model \"{model}\" on architecture \"{arch}\"...")

        # self.data[layer_name][num_banks]
        self.data: Dict[str, Dict[str, Dict[str, str]]] = file_data[model][arch] 
        
    def get_cost(self, layer_name: str, num_banks: int, recurrent: bool) -> int:
        if num_banks == 0:
            return MAX_VAL
        if recurrent:
            #HACK: 9 recurrent layers
            layer_name = mod_name_id(layer_name, 9)
        ret_str = self.data[layer_name][str(num_banks)]["performance"]
        if ret_str == "Infeasible":
            return MAX_VAL
        return int(self.data[layer_name][str(num_banks)]["performance"])