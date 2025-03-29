## OptiPIM Workload Folder

Each folder here represents a workload for OptiPIM. 

The compilation flow for a workload is as follows:
- We take a Pytorch model and specify input information (batch, channel, H, W)
- we use the `torch-mlir` package to compile the model's execution graph to MLIR.linalg format
- The OptiPIM compiler requires some additional info from each computation-heavy operator: currently they are conv2d and FC. You may refer to any `.linalg.mlir` file to understand its structure. 
  The added attributes are
  - num_banks: the maximum banks allowed for this operator to occupy
  - layer_group: to avoid runtime issues with solving MILP, we attack a large model by dividing the model into layer groups, and we individually solve for their MILP. Currently, this information comes from another dynamic programming solver and is manually parsed into the operators in the `.linalg.mlir` files.
  - global_layer_idx: layer id in the model
  - type_layer_idx: layer id for its type (e.g. `conv2D`)
- The compiler toolchain executable adds these two pieces of information using a pass. The produced files are named `layergroup.<workload>.linalg.mlir`
- Example
```mlir
%13 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, global_layer_idx = 2 : i32, layer_group = 0 : i32, num_banks = 58 : i32, strides = dense<1> : vector<2xi64>, type_layer_idx = 2 : i32}
ins(%padded_37, %cst_6 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%9 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
```
Note: for parsing convenience, right now, each layer is separated into its own MLIR file. 

Note: recently, there have been significant changes and instabilities with the torch-mlir package, and we are working towards other methods of lowering to MLIR. 

The OptiPIM optimization flow is as follows
- OptiPIM parses the processed `layerinfo.linalg.mlir` files for a workload to construct the MILP problem
  - NOTE: OptiPIM does affine nested loop-level transformations, but we do not rely on the affine dialect from MLIR (doing so introduces a lot of engineering issues).
    Instead, in MLIR sense, we only lower to the `linalg` dialect to expose the detailed information on the operators. 
    Then, we extract this information to construct the nested loop structure and loop bounds to construct the MILP problem
- OptiPIM solves the MILP problem and produces the transformed nested loop representation of the workload. This includes the loop bounds, loop order, and types of loop.
- The reconstructed nested loop is fed into the simulator to convert to traces.
- The simulator produces the results for such a transformation. 

### Adding new workloads

Any workload written in PyTorch and inheriting the `torch.nn.Module` class is technically supported by the `torch-mlir` toolchain to lower it down to the linalg dialect. 
However, you may encounter some problems with certain types of model structures and layers. For example, we had tremendous trouble converting BERT model. 

- https://github.com/llvm/torch-mlir
