import torch

import transformers
from typing import Any, Literal, Optional, Sequence, cast, overload, Tuple
from functools import partial

from .typing_mm import (
    Model,
) 
"""
    NOTE
    Attention Monitor for LVLMs, which can be used to get and visualize the attention matrix of selected layer 
    during the prefill process. The attention monitor is implemented as a callback function that can 
    be passed to the decoding function of the LVLM. 
    
    The attention monitor will save the attention matrix of selected layer at each decoding step, and 
    can be visualized.   
"""

class AttentionMonitor:
    def __init__(self, model, model_layers, layer_id_list):
        self.model = model
        self.model_layers = model_layers
        self.layer_id_list = layer_id_list
        self.attention_matrices = {layer_id: [] for layer_id in layer_id_list}
        self.generate_inputs = []  # 用于存储 generate 的输入参数
        self.hook_handles = []  # 用于存储注册的 hook 的句柄，以便后续移除
        self.generate_hook_handle = None # 用于存储 generate hook 的句柄
        self.original_generate = None # 用于存储原始的 generate 方法，以便后续恢复

    def attention_hook(self, 
                       module: torch.nn.Module, 
                       input,
                       output,
                       layer_id: int):
        """
            NOTE
            该hook函数挂载于 self_attn module 的 forward 函数上
            transformers中attention通常返回tuple, output[0]: attn_output, output[1]: attn_weights
        """
        
        bs = output[0].shape[0]
        seq_len = output[0].shape[1]
        dim = output[0].shape[2]
        # NOTE 只在 prefill 阶段获取 attention matrix
        attention_matrix = None
        if seq_len > 1:
            attention_matrix = output[1]
        if attention_matrix is not None:
            self.attention_matrices[layer_id].append(attention_matrix.detach().cpu())
        
    """
        NOTE
        挂载 attention_hook 函数的声明
    """
    def apply_attention_hooks(self):
        for layer_id in self.layer_id_list:
            layer_module = self.model_layers[layer_id]
            # 获取self attention模块
            if hasattr(layer_module, "self_attn"):
                attn_module = layer_module.self_attn
            elif hasattr(layer_module, "attention"):
                attn_module = layer_module.attention
            else:
                raise RuntimeError(f"Layer {layer_id} has no attention module")
            
            # attention_hook 函数挂载到 lvlm 模型指定的层的 self-attn module 上
            registered_attention_hook = attn_module.register_forward_hook(
                lambda module, input, output, layer_id=layer_id:
                self.attention_hook(module, input, output, layer_id)
            )
            self.hook_handles.append(registered_attention_hook)

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

        self.hook_handles = []
        self.remove_generate_hook()

    def generate_input_hook(self, args, kwargs):
        """
        Hook function to capture input parameters from model.generate()
        This captures both positional and keyword arguments passed to generate
        """
        # Store the inputs for later analysis
        self.generate_inputs.append({
            'args': args,
            'kwargs': kwargs
        })
        # You can also log or print the inputs here if needed
        # print(f"Generate called with args: {args}, kwargs keys: {kwargs.keys() if kwargs else None}")

    def apply_generate_hook(self):
        """
        Register the generate hook on the model
        """
        # Use register_forward_pre_hook to capture inputs before generate runs
        # or monkey patch the generate method
        original_generate = self.model.generate
        self.original_generate = original_generate  # Store original generate method

        def hooked_generate(*args, **kwargs):
            self.generate_input_hook(args, kwargs)
            return original_generate(*args, **kwargs)

        self.model.generate = hooked_generate

    def remove_generate_hook(self):
        """
        Restore the original generate method
        """
        if hasattr(self.model, 'generate') and callable(self.model.generate):
            self.model.generate = self.original_generate 

    


    
    

        