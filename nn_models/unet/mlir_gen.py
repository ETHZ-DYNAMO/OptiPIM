import torch
import torch.nn as nn

import torch_mlir

import torch_mlir.compiler_utils
import torch_mlir.torchscript


def main():
    batch_size = 1
    in_ch = 3
    H, W = 224, 224
    input_feature = torch.ones(batch_size, in_ch, H, W)
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
    
    for parameter in model.parameters():
        nn.init.ones_(parameter.data)
    
    model.eval()

    model_mlir = torch_mlir.torchscript.compile(model,
                                    input_feature,
                                    output_type=torch_mlir.compiler_utils.OutputType.LINALG_ON_TENSORS)

    model_mlir.dump()


if __name__ == "__main__":
    main()
