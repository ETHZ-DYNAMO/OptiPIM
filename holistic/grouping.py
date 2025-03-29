# This is the file for the algorithm of layer grouping using dynamic programming
from typing import List, Tuple
from database import Database
from argparse import ArgumentParser
from logging import Logger
import logging

def dp(n_layers : int, n_banks : int, group_bank_limit : int, group_layer_limit : int, cost : List[List[List[float]]], logger: Logger):
    # f[i][j]: minimum cost of using j banks for the first i layers
    f : List[List[float]] = [[float('inf') for j in range(n_banks+1)] for i in range(n_layers + 1)]
    # decision_group[i][j]: the bound of layer group of f[i][j]
    decision_group : List[List[int]] = [[-1 for j in range(n_banks+1)] for i in range(n_layers + 1)]
    # decision_bank[i][j]: the number of banks of layer group of f[i][j]
    decision_bank: List[List[int]] = [[-1 for j in range(n_banks+1)] for i in range(n_layers + 1)]
    # cost[i][j][k]: the cost of allocating k banks for layers[i:j]; previous cost[i][k]
    # cost[i][j][k] = baseline[i][k] + baseline[i+1][k] + ... + baseline[j][k]
    f[0][0] = 0.

    # [i:j] k: layer group using k banks
    
    for j in range(1, n_layers + 1):
        logger.debug(f"{j} / {n_layers}")
        for k1 in range(1, n_banks+1):
            start = max(1, j-group_layer_limit+1)
            for i in range(start, j+1):
                limit = min(group_bank_limit, k1)
                for k2 in range(1, limit+1):
                    f[j][k1] = min(f[j][k1], f[i-1][k1-k2] + cost[i][j][k2])
                    if f[j][k1] == f[i-1][k1-k2] + cost[i][j][k2]:
                        decision_group[j][k1] = i
                        decision_bank[j][k1] = k2
    logger.debug(f[n_layers][n_banks])

    # Infer decision
    groups: List[Tuple[int, int, int]] = []
    end_layer = n_layers
    cur_bank = n_banks
    while end_layer != 0 and cur_bank != 0:
        start_layer = decision_group[end_layer][cur_bank]
        used_bank = decision_bank[end_layer][cur_bank]
        groups.append((start_layer, end_layer, used_bank))
        logger.debug(f"{start_layer}, {end_layer}, {used_bank} - {cur_bank}")
        end_layer = start_layer - 1
        cur_bank -= used_bank
    logger.debug(groups)

    return groups

def get_layers(file: str, n_banks: int, database: Database, recurrent: bool) -> Tuple[int, List[List[int]]]:
    n_layers: int = 0
    layer_bank_cost : List[List[int]] = [[0] * (n_banks+1)]
    
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

    with open(file) as f:
        for line in f:
            n_layers += 1

            layer_name = get_layer_name(line)

            # all banks for a layer in a batch
            bank_cost = [database.get_cost(layer_name, i, recurrent) for i in range(n_banks+1)]
            layer_bank_cost.append(bank_cost)

    return n_layers, layer_bank_cost

def layer_group(file : str, n_banks : int, group_bank_limit : int, group_layer_limit : int, database: Database, recurrent: bool, logger: Logger) -> List[Tuple[int, int, int]]:
    """
    Args:
    file: the net file
    n_banks: the number of banks in the memory
    group_bank_limit: the maximum number of banks in a layer group
    group_layer_limit: the maximum number of layers in a layer group

    Returns:
    groups: a list of tuples, each tuple is a layer group, with the first element being the start layer, the second element being the end layer, and the third element being the number of banks used
    """
    n_layers, layer_cost = get_layers(file, 128, database, recurrent)

    cost : List[List[List[float]]] = []
    for i in range(0, n_layers + 1):
        cost.append([])
        for j in range(0, n_layers + 1):
            cost[i].append([])
            for k in range(0, group_bank_limit + 1):
                cost[i][j].append(0.)
    for i in range(0, n_layers + 1):
        for k in range(0, group_bank_limit + 1):
            sum = 0
            for j in range(i, n_layers + 1):
                sum += layer_cost[j][k]
                cost[i][j][k] = sum

    return dp(n_layers, n_banks, group_bank_limit, group_layer_limit, cost, logger)

if __name__ == "__main__":
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
    parser.add_argument(
        "--group_bank_limit", type=int, help="#Banks limit in the layer group"
    )
    parser.add_argument(
        "--group_layer_limit", type=int, help="#Layers limit in the layer group"
    )
    parser.add_argument(
        "--recurrent", action="store_true", help="Whether the model is recurrent"
    )
    args = parser.parse_args()

    # parse arguments
    database_file : str = args.database_file
    model_file : str = args.model_file
    model_name : str = args.model_name
    arch : str = args.arch
    n_banks : int = args.n_banks
    group_bank_limit : int = args.group_bank_limit
    group_layer_limit : int = args.group_layer_limit
    recurrent: bool = args.recurrent

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    database = Database(database_file, model_name, arch, logger)

    layer_group(model_file, n_banks, group_bank_limit, group_layer_limit, database, recurrent, logger)