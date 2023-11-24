import time

import torch
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
from setproctitle import setproctitle

from utils.prompter import Prompter


setproctitle("potatowook")

# change this to "nlpai-lab/kullm-polyglot-12.8b-v2" !!
MODEL = "/mnt/raid6/potatowook/kullm-enhance-ckpt/kullm-enhance1_3/checkpoint-3000"

model_16 = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

model_16.eval()

prompter = Prompter("kullm")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-12.8b")

def time_delta(fn, *args, **kwargs):
    start = time.time()
    res = fn(*args, **kwargs)
    end = time.time()
    print(f"elapsed time: {end - start}")
    return res


def infer(pipe, instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)
    return result


pipe_16 = pipeline("text-generation", model=model_16, tokenizer=tokenizer, device=0)
# pipe_32 = pipeline("text-generation", model=model_32, tokenizer=MODEL, device=1)

input_text = "python 언어와 javascript 언어의 차이점에 대해 설명해 줘."
res = time_delta(infer, pipe_16, input_text=input_text)
print(res)
print("---")
input_text = "한국의 아이돌 그룹 몬스타엑스에 대해 알려줘."
res = time_delta(infer, pipe_16, input_text=input_text)
print(res)
print("---")
input_text = "야 심심한데 재롱이나 함 피워 봐라 "
res = time_delta(infer, pipe_16, input_text=input_text)
print(res)
print("---")
input_text = "강릉 초당두부마을에서 가장 유명한 짬뽕순두부집의 이름은?"
res = time_delta(infer, pipe_16, input_text=input_text)
print(res)
print("---")