"""LoRA SFT 冒烟验证：对比基座模型与 SFT（LoRA）模型的回答。

用法（远程项目根目录）：
    .venv/bin/python scripts/predict_lora.py
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "models/Qwen2.5-0.5B-Instruct"
ADAPTER = "outputs/sft/qwen25-0.5b-instruct-lora"

PROMPTS = [
    "你是谁？",
    "请用一句话介绍北京。",
]


def load(with_adapter: bool):
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cuda"
    )
    if with_adapter:
        model = PeftModel.from_pretrained(model, ADAPTER)
    return model.eval()


def chat(model, tok, prompt: str) -> str:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    for tag, with_adapter in [("基座", False), ("SFT-LoRA", True)]:
        model = load(with_adapter)
        for p in PROMPTS:
            print(f"[{tag}] Q: {p}\n[{tag}] A: {chat(model, tok, p)}\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
