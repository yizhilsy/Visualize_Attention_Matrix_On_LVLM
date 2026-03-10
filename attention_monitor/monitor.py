import torch
import transformers
from typing import Any, Literal, Optional, Sequence, cast, overload, Tuple
from functools import partial
import os
import json
import uuid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
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
    def __init__(self, model, model_layers, layer_id_list, tokenizer, image_processor):
        self.model = model
        self.model_layers = model_layers
        self.layer_id_list = layer_id_list
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self.attention_matrices = {layer_id: [] for layer_id in layer_id_list}
        self.image_position_masks = [] # 用于存储推理时每个 batch 的 image position mask
        self.hook_handles = []  # 用于存储注册的 hook 的句柄，以便后续移除
        self.generate_hook_handle = None # 用于存储 generate hook 的句柄
        self.original_generate = None # 用于存储原始的 generate 方法，以便后续恢复
        self.vision_token_attention_scores = {layer_id: [] for layer_id in layer_id_list} # 用于存储每个层的视觉token的注意力分数，值为列表
        self.image_tensor_list: torch.Tensor = []

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
        # 单次存入一张图片的张量
        for img in images:
            self.image_tensor_list.append(img.detach().cpu())
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
    
    def plot_attention_heatmap(self):
        pass


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
        PART B
        基于 self.layer_id_list 中指定的层数、self.attention_matrices 中记录的 attention matrix 和
        self.image_position_masks 中记录的 image position mask
        可视化每个层数的 vision token 的注意力分数
        """

        print("\n=== VISION TOKEN ATTENTION SCORE ANALYSIS ===")

        # 遍历所有层
        for layer_id in self.layer_id_list:
            layer_attention_matrices = self.attention_matrices[layer_id]

            print(f"\nLayer {layer_id}:")

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

                    # 获取视觉token的索引
                    vision_indices = torch.where(sample_image_mask)[0]

                    if len(vision_indices) == 0:
                        continue

                    # 对每个注意力头，计算每个image token作为key时的注意力分数总和
                    sample_vision_scores = []

                    for head_idx in range(num_heads):
                        head_attention = sample_attention[head_idx]  # shape: (seq_len, seq_len)

                        # 对每个image token，累加其作为key时的注意力分数（即对head_attention的各列求和）
                        head_vision_scores = []
                        for vision_idx in vision_indices:
                            # 累加该image token在所有query位置上的注意力分数
                            vision_score = head_attention[:, vision_idx].sum().item()
                            head_vision_scores.append(vision_score)

                        sample_vision_scores.append(head_vision_scores)

                    # 在注意力头粒度上进行平均
                    # sample_vision_scores: (num_heads, num_vision_tokens)
                    sample_vision_scores = torch.tensor(sample_vision_scores)  # 转换为tensor便于计算
                    avg_vision_scores = sample_vision_scores.mean(dim=0)  # (num_vision_tokens,)

                    # 存储到成员变量中
                    self.vision_token_attention_scores[layer_id].append({
                        'avg_attention_scores': avg_vision_scores.tolist(),
                        'num_vision_tokens': len(vision_indices),
                        'num_heads': num_heads
                    })

                    # 打印统计信息
                    print(f"  Batch {batch_idx}, Sample {sample_idx}:")
                    print(f"    Number of vision tokens: {len(vision_indices)}")
                    print(f"    Average vision token scores: {avg_vision_scores.mean():.4f}")
                    print(f"    Max vision token score: {avg_vision_scores.max():.4f}")
                    print(f"    Min vision token score: {avg_vision_scores.min():.4f}")

        # 保存vision token注意力分数到文件
        if hasattr(self, 'vision_token_attention_scores') and self.vision_token_attention_scores:
            vision_scores_file = os.path.join(result_folder, "vision_token_attention_scores.json")
            with open(vision_scores_file, 'w', encoding='utf-8') as f:
                # 将tensor转换为list以便JSON序列化
                serializable_scores = {}
                for layer_id, samples in self.vision_token_attention_scores.items():
                    serializable_scores[str(layer_id)] = samples
                json.dump(serializable_scores, f, ensure_ascii=False, indent=2)
            print(f"\nVision token attention scores saved to: {vision_scores_file}")

        """
        PART C
        基于 image patch 细粒度生成每个 sample 中的图片在指定 layer 上的可视化结果
        将 vision token 的重要性分数映射回原始图像的热力图
        """
        print("\n=== GENERATING ATTENTION HEATMAP VISUALIZATIONS ===")

        # 检查是否存在必要的成员变量
        if not hasattr(self, 'image_tensor_list') or not self.image_tensor_list:
            print("Warning: No image_tensor_list found. Skipping heatmap generation.")
            return

        if not hasattr(self, 'vision_token_attention_scores') or not self.vision_token_attention_scores:
            print("Warning: No vision_token_attention_scores found. Skipping heatmap generation.")
            return

        # 创建结果文件夹
        heatmap_dir = os.path.join(result_folder, "attn_heatmap")
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)

        print(f"Heatmap results will be saved to: {heatmap_dir}")

        # 遍历每个层
        for layer_id in self.layer_id_list:
            if not self.vision_token_attention_scores[layer_id]:
                print(f"Warning: No vision token attention scores found for layer {layer_id}.")
                continue

            layer_samples = self.vision_token_attention_scores[layer_id]

            # 遍历每个样本
            for sample_idx, sample in enumerate(tqdm(layer_samples, desc=f"Processing layer {layer_id}")):
                avg_scores = sample['avg_attention_scores']
                num_vision_tokens = sample['num_vision_tokens']

                if not avg_scores or num_vision_tokens == 0:
                    continue

                # 获取对应的原始图像
                if sample_idx >= len(self.image_tensor_list):
                    print(f"Warning: No image found for sample {sample_idx}.")
                    continue

                original_image = self.image_tensor_list[sample_idx]  # 形状: (3, h, w)
                if isinstance(original_image, torch.Tensor):
                    original_image = original_image.cpu().numpy()
                    original_image = np.transpose(original_image, (1, 2, 0))  # 转换为 (h, w, 3)

                original_image = original_image.astype(np.float32)

                # 如果是normalize后的图像 (如ViT/LLaVA)
                imagenet_mean = self.image_processor.image_mean
                imagenet_std = self.image_processor.image_std
                if original_image.min() < 0:  # 说明做过normalize
                    original_image = original_image * imagenet_std + imagenet_mean
                # 如果仍是0~255
                if original_image.max() > 1.5:
                    original_image = original_image / 255.0
                original_image = np.clip(original_image, 0, 1)

                # 创建样本专属的文件夹
                sample_dir = os.path.join(heatmap_dir, f"sample_{sample_idx}")
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)

                # 动态计算最佳的网格尺寸
                # 尝试找到最接近的平方根作为网格尺寸
                sqrt_num_tokens = int(np.sqrt(num_vision_tokens))
                grid_sizes = []

                # 生成可能的网格尺寸（考虑常见的图像patch排列）
                for i in range(1, sqrt_num_tokens + 5):
                    for j in range(1, sqrt_num_tokens + 5):
                        if i * j == num_vision_tokens:
                            grid_sizes.append((i, j))

                # 如果没有找到精确的匹配，选择最接近的网格
                if not grid_sizes:
                    # 使用最接近的平方根作为近似
                    grid_size = sqrt_num_tokens
                    if grid_size * grid_size < num_vision_tokens:
                        grid_size += 1
                    grid_size = (grid_size, grid_size)
                    print(f"Warning: Cannot find exact grid for {num_vision_tokens} tokens, using {grid_size[0]}x{grid_size[1]} approximation.")
                else:
                    # 选择最接近正方形的网格
                    grid_sizes.sort(key=lambda x: abs(x[0] - x[1]))
                    grid_size = grid_sizes[0]  # 保持为元组 (height, width)
                    print(f"Sample {sample_idx}: Found grid size {grid_size} for {num_vision_tokens} tokens")

                # 创建2D注意力图
                attention_map = np.zeros(grid_size)

                # 填充注意力分数
                for idx, score in enumerate(avg_scores):
                    if idx >= grid_size[0] * grid_size[1]:
                        break
                    row = idx // grid_size[1]
                    col = idx % grid_size[1]
                    attention_map[row, col] = score

                # 调整注意力图的大小以匹配原始图像
                img_h, img_w = original_image.shape[:2]
                attention_map_resized = np.zeros((img_h, img_w))

                # 使用插值调整大小，插值方法为双线性插值
                from scipy.ndimage import zoom
                y_scale = img_h / grid_size[0]
                x_scale = img_w / grid_size[1]
                attention_map_resized = zoom(attention_map, (y_scale, x_scale), order=1)

                # 归一化注意力图
                attention_map_resized = (attention_map_resized - np.min(attention_map_resized)) / \
                                      (np.max(attention_map_resized) - np.min(attention_map_resized) + 1e-8)

                # 创建热力图
                heatmap = plt.cm.viridis(attention_map_resized)  # 使用viridis颜色映射
                heatmap = heatmap[:, :, :3]  # 移除alpha通道

                # 确保heatmap的数据类型正确
                heatmap = heatmap.astype(np.float32)

                # 将热力图叠加到原始图像上
                alpha = 0.6  # 热力图的透明度
                overlay = (1 - alpha) * original_image + alpha * heatmap

                # 确保overlay的数据类型正确
                overlay = overlay.astype(np.float32)
                overlay = np.clip(overlay, 0.0, 1.0)

                # 为显示准备图像副本
                display_original = (original_image * 255).astype(np.uint8)

                # 创建对比图：原始图像、热力图、叠加图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # 原始图像
                axes[0].imshow(display_original)
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                # 热力图
                im = axes[1].imshow(attention_map_resized, cmap='viridis')
                axes[1].set_title('Attention Heatmap')
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                # 叠加图 - 转换为uint8用于显示
                display_overlay = (overlay * 255).astype(np.uint8)
                axes[2].imshow(display_overlay)
                axes[2].set_title('Overlay')
                axes[2].axis('off')

                plt.suptitle(f'Layer {layer_id} - Sample {sample_idx}', fontsize=16)
                plt.tight_layout()

                # 保存图像
                output_path = os.path.join(sample_dir, f"layer_{layer_id}_heatmap.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                plt.close()

                # 单独保存叠加图（更高分辨率）
                plt.figure(figsize=(10, 10))
                plt.imshow(overlay)
                plt.title(f'Layer {layer_id} - Sample {sample_idx}', fontsize=16)
                plt.axis('off')
                plt.savefig(os.path.join(sample_dir, f"layer_{layer_id}_overlay.png"),
                           dpi=300, bbox_inches='tight', pad_inches=0.1)
                plt.close()

                print(f"  Sample {sample_idx} layer {layer_id} heatmap saved")

        print(f"\nAll heatmap visualizations saved to: {heatmap_dir}")
        



