import torch
import transformers
from typing import Any, Literal, Optional, Sequence, cast, overload, Tuple
from functools import partial
import os
import json
import uuid
from .typing_mm import (
    Model,
)
from patchs_for_model.llava import prepare_inputs_image_position_mask_for_multimodal

"""
    NOTE
    Attention Monitor for LVLMs, which can be used to get and visualize the attention matrix of selected layer 
    during the prefill process. The attention monitor is implemented as a callback function that can 
    be passed to the decoding function of the LVLM. 
    
    The attention monitor will save the attention matrix of selected layer at each decoding step, and 
    can be visualized.   
"""

class AttentionMonitor:
    def __init__(self, model, model_layers, layer_id_list, tokenizer):
        self.model = model
        self.model_layers = model_layers
        self.layer_id_list = layer_id_list
        self.tokenizer = tokenizer

        self.attention_matrices = {layer_id: [] for layer_id in layer_id_list}
        self.image_position_masks = [] # 用于存储推理时每个 batch 的 image position mask
        self.hook_handles = []  # 用于存储注册的 hook 的句柄，以便后续移除
        self.generate_hook_handle = None # 用于存储 generate hook 的句柄
        self.original_generate = None # 用于存储原始的 generate 方法，以便后续恢复
        self.vision_token_attention_scores = {layer_id: [] for layer_id in layer_id_list} # 用于存储每个层的视觉token的注意力分数，值为列表

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

    """
        NOTE
        移除 attention_hook 以及 generate_hook
    """
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
        input_ids = args[0].clone()
        images = kwargs['images'].clone()
        # Create attention_mask: True for non-pad tokens, False for pad tokens
        attention_mask = (input_ids != self.tokenizer.pad_token_id)
        
        _, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_position_mask = prepare_inputs_image_position_mask_for_multimodal(
            self=self.model,
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            images=images,
        )
        self.image_position_masks.append(image_position_mask.detach().cpu())



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
    
    def statistic_and_visualize(self):
        """
        NOTE
        统计并可视化 attention matrix 的函数，功能如下:
        1. 基于注意力分数矩阵统计记录的层数的不同模态的注意力总分（贡献）
        2. 从 image patch 尺度上可视化记录的层数的各个视觉token的注意力分数
        """

        """
        PART A
        Statistical analysis of attention score contributions from different modalities
        """
        # 1. 数据验证
        if not self.attention_matrices or not self.image_position_masks:
            raise ValueError("No attention matrices or image position masks found. Please run inference first.")

        # 2. 初始化统计结果存储结构
        # 结构: {layer_id: {'vision_to_vision': [], 'vision_to_text': [], 'text_to_vision': [], 'text_to_text': []}}
        statistics = {layer_id: {
            'vision_to_vision': [],
            'vision_to_text': [],
            'text_to_vision': [],
            'text_to_text': []
        } for layer_id in self.layer_id_list}

        # 3. 遍历所有层
        for layer_id in self.layer_id_list:
            layer_attention_matrices = self.attention_matrices[layer_id]

            # 遍历每个batch的注意力矩阵
            for batch_idx, attention_matrix in enumerate(layer_attention_matrices):
                # attention_matrix shape: (bs, attn_head, seq_len, seq_len)
                bs, num_heads, seq_len, _ = attention_matrix.shape

                # 获取对应的image position mask
                if batch_idx >= len(self.image_position_masks):
                    continue

                image_position_mask = self.image_position_masks[batch_idx]  # shape: (bs, seq_len)

                # 处理每个batch中的每个样本
                for sample_idx in range(bs):
                    sample_attention = attention_matrix[sample_idx]  # shape: (num_heads, seq_len, seq_len)
                    sample_image_mask = image_position_mask[sample_idx]  # shape: (seq_len,)

                    # 创建文本token的掩码
                    text_mask = ~sample_image_mask

                    # 获取视觉和文本token的索引
                    vision_indices = torch.where(sample_image_mask)[0]
                    text_indices = torch.where(text_mask)[0]

                    # 如果没有视觉或文本token，跳过
                    has_vision = len(vision_indices) > 0
                    has_text = len(text_indices) > 0
                    if not has_vision or not has_text:
                        continue

                    # 计算每种注意力模式的平均分数（对每个头分别计算）
                    head_avg_scores = {
                        'vision_to_vision': [],
                        'vision_to_text': [],
                        'text_to_vision': [],
                        'text_to_text': []
                    }

                    for head_idx in range(num_heads):
                        head_attention = sample_attention[head_idx]  # shape: (seq_len, seq_len)

                        # 1. vision → vision
                        if has_vision:
                            v2v_attention = head_attention[vision_indices][:, vision_indices]
                            v2v_sum = v2v_attention.sum().item() if v2v_attention.numel() > 0 else 0.0
                            head_avg_scores['vision_to_vision'].append(v2v_sum)

                        # 2. vision → text
                        if has_vision and has_text:
                            v2t_attention = head_attention[vision_indices][:, text_indices]
                            v2t_sum = v2t_attention.sum().item() if v2t_attention.numel() > 0 else 0.0
                            head_avg_scores['vision_to_text'].append(v2t_sum)

                        # 3. text → vision
                        if has_text and has_vision:
                            t2v_attention = head_attention[text_indices][:, vision_indices]
                            t2v_sum = t2v_attention.sum().item() if t2v_attention.numel() > 0 else 0.0
                            head_avg_scores['text_to_vision'].append(t2v_sum)

                        # 4. text → text
                        if has_text:
                            t2t_attention = head_attention[text_indices][:, text_indices]
                            t2t_sum = t2t_attention.sum().item() if t2t_attention.numel() > 0 else 0.0
                            head_avg_scores['text_to_text'].append(t2t_sum)

                    # 对所有注意力头取平均
                    statistics[layer_id]['vision_to_vision'].append(
                        sum(head_avg_scores['vision_to_vision']) / num_heads if head_avg_scores['vision_to_vision'] else 0.0
                    )
                    statistics[layer_id]['vision_to_text'].append(
                        sum(head_avg_scores['vision_to_text']) / num_heads if head_avg_scores['vision_to_text'] else 0.0
                    )
                    statistics[layer_id]['text_to_vision'].append(
                        sum(head_avg_scores['text_to_vision']) / num_heads if head_avg_scores['text_to_vision'] else 0.0
                    )
                    statistics[layer_id]['text_to_text'].append(
                        sum(head_avg_scores['text_to_text']) / num_heads if head_avg_scores['text_to_text'] else 0.0
                    )

        # 4. 整理最终结果
        # 转换为更友好的格式: {layer_id: {mode: [batch_scores]}}
        final_statistics = {}
        for layer_id in self.layer_id_list:
            if layer_id not in final_statistics:
                final_statistics[layer_id] = {}

            for mode in ['vision_to_vision', 'vision_to_text', 'text_to_vision', 'text_to_text']:
                scores = statistics[layer_id][mode]
                final_statistics[layer_id][mode] = scores

        # 打印统计信息
        total_samples = sum(len(scores) for layer_stats in final_statistics.values() for scores in layer_stats.values()) // 4
        print(f"Total samples processed: {total_samples}")

        # 初始化汇总统计信息
        # 结构: {layer_id: {'vision_to_vision': total_score, 'vision_to_text': total_score, ...}}
        summary_stats = {
            layer_id: {
                'vision_to_vision': 0.0,
                'vision_to_text': 0.0,
                'text_to_vision': 0.0,
                'text_to_text': 0.0,
                'text_modality': 0.0,
                'vision_modality': 0.0
            }
            for layer_id in self.layer_id_list
        }

        # 在各层各个模式中计算所有样例上的注意力总分
        for layer_id, layer_stats in final_statistics.items():
            print(f"Layer {layer_id}:")

            for mode, scores in layer_stats.items():
                mode_total = sum(scores)
                mode_avg = mode_total / len(scores) if scores else 0.0
                mode_max = max(scores) if scores else 0.0
                mode_min = min(scores) if scores else 0.0

                print(f"  {mode}:")
                print(f"    Total Attention Score = {mode_total:.4f} over {len(scores)} samples")
                print(f"    Average Score = {mode_avg:.4f}")
                print(f"    Max Score = {mode_max:.4f}")
                print(f"    Min Score = {mode_min:.4f}")

                # 累加到该层的对应模式总分
                summary_stats[layer_id][mode] = mode_total
                summary_stats[layer_id]['text_modality'] = summary_stats[layer_id]['vision_to_text'] + summary_stats[layer_id]['text_to_text']
                summary_stats[layer_id]['vision_modality'] = summary_stats[layer_id]['vision_to_vision'] + summary_stats[layer_id]['text_to_vision']

        # 计算总体统计（跨所有层）
        overall_totals = {
            'vision_to_vision': sum(summary_stats[layer_id]['vision_to_vision'] for layer_id in self.layer_id_list),
            'vision_to_text': sum(summary_stats[layer_id]['vision_to_text'] for layer_id in self.layer_id_list),
            'text_to_vision': sum(summary_stats[layer_id]['text_to_vision'] for layer_id in self.layer_id_list),
            'text_to_text': sum(summary_stats[layer_id]['text_to_text'] for layer_id in self.layer_id_list),
            'text_modality': sum(summary_stats[layer_id]['text_modality'] for layer_id in self.layer_id_list),
            'vision_modality': sum(summary_stats[layer_id]['vision_modality'] for layer_id in self.layer_id_list)
        }

        # 打印总体统计
        print("\n=== OVERALL STATISTICS (ACROSS ALL LAYERS) ===")
        for mode, total in overall_totals.items():
            print(f"{mode}: Total = {total:.4f}")

        # 5. 保存结果到文件
        # 创建结果文件夹
        results_dir = "./results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # 生成随机uid作为文件夹名称
        uid = str(uuid.uuid4())
        result_folder = os.path.join(results_dir, uid)
        os.makedirs(result_folder)

        # 保存详细的汇总统计信息到JSON文件
        summary_file = os.path.join(result_folder, "summary_statistics.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)

        # 保存更详细的统计信息（包含每个样本的原始分数）
        detailed_stats = {
            'summary': summary_stats,
            'raw_data': final_statistics,
            'metadata': {
                'total_samples': total_samples,
                'num_layers': len(self.layer_id_list),
                'layer_ids': list(self.layer_id_list)
            }
        }
        detailed_file = os.path.join(result_folder, "detailed_statistics.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_stats, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {result_folder}")
        print(final_statistics)
    
        """
        TODO
        PART B
        基于 self.layer_id_list 中指定的层数、self.attention_matrices 中记录的 attention matrix 和
        self.image_position_masks 中记录的 image position mask
        可视化每个层数的 vision token 的注意力分数
        """
        
        



