import torch
import torch.nn as nn

def create_model(config, input_shape):
    """
    Creates a PyTorch model from a configuration dictionary.
    
    Args:
        config (dict): A dictionary with a 'model' key containing a 'layers' list.
                       Each layer is a dict specifying the type and parameters.
        input_shape (tuple): The shape of the input in the form (channels, height, width).
                             For example, (3, 32, 32) for a 32x32 RGB image.
    
    Returns:
        nn.Module: A PyTorch model built as a sequential container.
    """
    layers_config = config['model']['layers']
    modules = []
    
    current_shape = input_shape
    
    for layer in layers_config:
        layer_type = layer['type']
        
        if layer_type == 'conv2d':
            if not isinstance(current_shape, tuple) or len(current_shape) != 3:
                raise ValueError("conv2d layer expects a 3D input shape (channels, height, width)")
            in_channels = current_shape[0]
            out_channels = layer['filters']
            kernel_size = layer['kernel_size']

            conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            modules.append(conv)

            _, H, W = current_shape
            H_new = H - kernel_size + 1
            W_new = W - kernel_size + 1
            current_shape = (out_channels, H_new, W_new)
            
        elif layer_type == 'relu':
            modules.append(nn.ReLU())
            
        elif layer_type == 'max_pool2d':
            kernel_size = layer['kernel_size']
            pool = nn.MaxPool2d(kernel_size)
            modules.append(pool)
            if not isinstance(current_shape, tuple) or len(current_shape) != 3:
                raise ValueError("max_pool2d layer expects a 3D input shape (channels, height, width)")
            C, H, W = current_shape

            H_new = H // kernel_size
            W_new = W // kernel_size
            current_shape = (C, H_new, W_new)
            
        elif layer_type == 'flatten':
            modules.append(nn.Flatten())
            if isinstance(current_shape, tuple):
                C, H, W = current_shape
                current_shape = C * H * W
            
        elif layer_type == 'linear':
            if not isinstance(current_shape, int):
                raise ValueError("linear layer expects flattened input. Insert a 'flatten' layer before a 'linear' layer.")
            in_features = current_shape
            out_features = layer['units']
            linear = nn.Linear(in_features, out_features)
            modules.append(linear)

            current_shape = out_features
            
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
    
    model = nn.Sequential(*modules)
    return model