import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModelForCausalLM, PeftConfig

from utils.prompter import Prompter

MODEL = "/data/potatowook/hclt/ckpt/baseline3/checkpoint-2385"
original_model = "EleutherAI/polyglot-ko-1.3b"

model = AutoModelForCausalLM.from_pretrained(
    original_model,
    # torch_dtype=torch.float16,
    # low_cpu_mem_usage=True,
)
model = PeftModelForCausalLM.from_pretrained(model, MODEL).to("cuda:0")
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=original_model, device=0)

prompter = Prompter("kullm")


def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result


while True:
    input_str = input("input: ")
    if "종료" in input_str:
        break
    result = infer(instruction=input_str)
    print(result)
