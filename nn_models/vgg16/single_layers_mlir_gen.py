import torch
import torch.nn as nn

import torch_mlir
from torchinfo import summary, ModelStatistics

import torch_mlir.compiler_utils
import torch_mlir.torchscript

from torchvision.models import vgg16


def collect_layers(model):
    layers = []
    for module in model.children():
        if isinstance(module, nn.Sequential):
            layers.extend(collect_layers(module))
        else:
            layers.append(module)
    return layers


def main():
    batch_size = 1
    in_ch = 3
    H, W = 224, 224
    model = vgg16()  
    
    model_stats: ModelStatistics = summary(model, input_size=(batch_size, in_ch, H, W), col_names=["input_size", "output_size"])

    single_layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
    conv_fc_layers = [layer for layer in single_layers if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)]
    
    
    conv_idx = 0
    fc_idx = 0
    global_idx = 0
        
    for layerinfo in model_stats.summary_list:
        
        if isinstance(layerinfo.module, nn.Conv2d):
            layer = layerinfo.module
            for parameter in model.parameters():
                nn.init.ones_(parameter.data)
            
            layer_input = torch.ones(layerinfo.input_size)
            layer_mlir = torch_mlir.torchscript.compile(
                layer,
                layer_input,
                output_type=torch_mlir.compiler_utils.OutputType.LINALG_ON_TENSORS
            )
            
            filename = f"single_layers/vgg16.l{global_idx}.conv{conv_idx}.linalg.mlir"
            print(f"Generate {filename}")
            with open(filename, "w") as f:
                f.write(str(layer_mlir))
            
            conv_idx += 1
            global_idx += 1

        if isinstance(layerinfo.module, nn.Linear):
            layer = layerinfo.module
            for parameter in model.parameters():
                nn.init.ones_(parameter.data)
            
            layer_input = torch.ones(layerinfo.input_size)
            layer_mlir = torch_mlir.torchscript.compile(
                layer,
                layer_input,
                output_type=torch_mlir.compiler_utils.OutputType.LINALG_ON_TENSORS
            )
            
            filename = f"single_layers/vgg16.l{global_idx}.fc{fc_idx}.linalg.mlir"
            print(f"Generate {filename}")
            with open(filename, "w") as f:
                f.write(str(layer_mlir))
            
            fc_idx += 1
            global_idx += 1

    # verify not missing layers
    assert(global_idx == len(conv_fc_layers))

if __name__ == "__main__":
    main()
