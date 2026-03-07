import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import AutoTokenizer

from attention_monitor.monitor import AttentionMonitor

# Initialize pre-trained LlaVA MLLM
model_path = "/s/models/llava-series/llava-v1.5-7b"
device = "cuda:0"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    use_flash_attn=False,
    device="cuda",
    device_map=device,
    attn_implementation="eager",
)

# 调用类方法挂载 hook 函数到模型指定层上
attention_monitor = AttentionMonitor(
    model=model,
    model_layers=model.model.layers,
    layer_id_list=[31],
    tokenizer=tokenizer,
)
attention_monitor.apply_attention_hooks()
attention_monitor.apply_generate_hook()

from datasets import load_dataset
dataset = load_dataset(
    "parquet",
    data_files="/home/lsy/shared_data/MME/data/*.parquet",
    split="train"
)

question = DEFAULT_IMAGE_TOKEN + '\n' + dataset[2]['question']
image = dataset[2]['image'].convert('RGB')
conv = conv_templates['vicuna_v1'].copy()
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
image_tensor = process_images([image], image_processor, model.config)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
input_ids = input_ids.to(device='cuda', non_blocking=True)

# model inference
with torch.inference_mode():
    output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image.size,
                temperature=0,
                max_new_tokens=128,
                use_cache=True)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

# 卸载 hook 函数
attention_monitor.remove_hooks()