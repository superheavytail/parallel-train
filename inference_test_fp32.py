"""Deepspeed의 zero_to_fp32.py 를 이용해서 아예 전체 모델을 불러오는 방식의 인퍼런스"""
"""Deepspeed가 이상하게 저장하기 때문에 이 코드는 사용하면 안 됨"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModelForCausalLM, PeftConfig, get_peft_model, LoraConfig

from utils.prompter import Prompter

MODEL = "/mnt/radi6/potatowook/hclt/ckpt/baseline2/checkpoint-7155/"
original_model = "EleutherAI/polyglot-ko-5.8b"

model = PeftModelForCausalLM.from_pretrained(
    original_model,
    # torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,
)

lora_r = 8
lora_alpha = 16
lora_target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
lora_dropout = 0.05
config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=['embed_out'],
)
model = get_peft_model(model, config)

PeftModelForCausalLM.from_pretrained()
state_dict = torch.load("/mnt/raid6/potatowook/hclt/ckpt/baseline2/checkpoint-7155/pytorch_model.bin")
# model = PeftModelForCausalLM.from_pretrained(model, MODEL).to("cuda:0")
model.load_state_dict(state_dict)
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=original_model, device=0)

prompter = Prompter("kullm")


def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result


result = infer(input_text="고려대학교에 대해서 알려줘")
print(result)