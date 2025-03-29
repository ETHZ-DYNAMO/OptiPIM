from __future__ import annotations
from argparse import ArgumentParser
from typing import List
from grouping import layer_group
from database import Database, MAX_VAL
from logging import Logger
import logging

def get_layer_name(line: str) -> str:
    line_str = line.strip().replace("\t", ",")
    tokens = [token.strip() for token in line_str.split(",")]
    layer_type = tokens[1]
    layer_id: str = tokens[2]

    if layer_type == "CONV2d":
        return f"conv{layer_id}"
    elif layer_type == "FC":
        return f"fc{layer_id}"
    elif layer_type == "batch_matmul":
        return f"fc{layer_id}"
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

def get_layer_names(file: str) -> List[str]:
    layer_names = []
    with open(file) as f:
        for line in f:
            layer_name = get_layer_name(line)
            layer_names.append(layer_name)
    return layer_names

def calculate_total_cycles(net_file: str, n_all_banks: int, database: Database, grouping: bool, recurrent: bool, logger: Logger) -> int:
    # Load the net.csv file
    layer_names = get_layer_names(net_file)
    num_layers = len(layer_names)

    if not grouping:
        num_bank_layer = n_all_banks // num_layers
        n_banks: List[int] = [num_bank_layer for _ in range(num_layers)]
    else:
        n_banks: List[int] = [-1 for _ in range(num_layers)]
        groups = layer_group(net_file, n_all_banks, 128, 1, database, recurrent, logger)
        for group in groups:
            start, end, n_bank = group
            for i in range(start - 1, end):
                logger.debug(f"Layer {i} assigned to bank {n_bank}")
                logger.debug(f"len(n_banks): {len(n_banks)}")
                n_banks[i] = n_bank
        assert -1 not in n_banks, "Some layers are not assigned to a bank"
    
    assert len(n_banks) == num_layers, f"Number of layers {num_layers} does not match number of n_banks {len(n_banks)}"
    
    layer_cycles = [database.get_cost(layer, n_bank, recurrent) for (layer, n_bank) in zip(layer_names, n_banks)]

    total_cycles = sum(layer_cycles)

    infeasible: bool = False
    for (name, cycle) in zip(layer_names, layer_cycles):
        if cycle == MAX_VAL:
            infeasible = True
            logger.debug(f"{name}: Infeasible")
        else:
            logger.debug(f"{name}: {cycle}")

    if infeasible:
        return MAX_VAL
    else:
        return total_cycles

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser
    parser.add_argument(
        "--database_file", type=str, help="Path to the database.json file"
    )
    parser.add_argument(
        "--model_file", type=str, help="Path to the model file"
    )
    parser.add_argument(
        "--model_name", type=str, help="Name of the model"
    )
    parser.add_argument(
        "--arch", type=str, help="The PIM architecture"
    )
    parser.add_argument(
        "--n_banks", type=int, help="#Banks in the memory"
    )
    parser.add_argument('--layer_grouping', action="store_true", help='Whether to use layer grouping', default=False)
    parser.add_argument('--recurrent', action="store_true", help='Whether to use recurrent layers', default=False)
    args = parser.parse_args()

    # parse arguments
    database_file : str = args.database_file
    model_file : str = args.model_file
    model_name : str = args.model_name
    arch : str = args.arch
    n_banks : int = args.n_banks
    grouping: bool = args.layer_grouping
    recurrent: bool = args.recurrent

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("one_case")

    database = Database(database_file, model_name, arch, logger)

    total_cycles = calculate_total_cycles(model_file, n_banks, database, grouping, recurrent, logger)

    if total_cycles == MAX_VAL:
        logger.debug("The model is infeasible")
    else:
        logger.debug(f"total cycles: {total_cycles}")
