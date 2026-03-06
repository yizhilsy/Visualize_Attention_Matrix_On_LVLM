from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers import AutoTokenizer

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