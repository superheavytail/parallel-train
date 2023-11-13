import random

import fire
import torch
from torch.nn.functional import log_softmax
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModelForCausalLM, PeftConfig
from datasets import load_dataset
from tqdm import tqdm
from setproctitle import setproctitle

from utils.prompter import Prompter


def calc_log_p(model, inp_tokens, cont_tokens):
    assert len(inp_tokens.shape) == 2 and inp_tokens.shape[0] == 1
    model_input = torch.cat((inp_tokens, cont_tokens), dim=-1).to(model.device)
    with torch.no_grad():
        output = model(model_input)

    logits = log_softmax(output[0], -1).to("cpu")
    cont_tokens = cont_tokens.to("cpu")

    inp_len = len(inp_tokens[0])
    cont_len = len(cont_tokens[0])
    cont_logits = logits[0][inp_len:]

    g = torch.gather(cont_logits, 1, cont_tokens)
    log_p = -torch.sum(g, dim=-1).item()
    return log_p  # 높을수록 cont_tokens를 생성할 확률이 높음!


def main(mode: str = None):
    if 'my' in mode:
        MODEL = "/data/potatowook/hclt/ckpt/my4/checkpoint-8022"
    elif 'baseline' in mode:
        MODEL = "/data/potatowook/hclt/ckpt/baseline4/checkpoint-7155"
    else:
        raise NotImplementedError
    original_model = "EleutherAI/polyglot-ko-1.3b"

    model = AutoModelForCausalLM.from_pretrained(
        original_model,
        # torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(original_model)
    model = PeftModelForCausalLM.from_pretrained(model, MODEL).to("cuda:0")
    model.eval()

    prompter = Prompter("kullm")

    instruction = "삼원색에 대해 설명해주세요."
    input_text = ""
    label = "세 가지 기본 색은 빨강, 파랑, 노랑입니다. 이 색은 다른 색을 혼합하여 만들 수 없고 다른 모든 색은 다양한 비율로 조합하여 만들 수 있기 때문에 원색이라고 부릅니다. 빛에 사용되는 첨가제 색상 시스템에서 원색은 빨강, 녹색, 파랑(RGB)입니다."
    prompt = prompter.generate_prompt(instruction, input_text, label)

    # test
    inp_tokens = tokenizer.encode(instruction, return_tensors='pt')
    cont_tokens = tokenizer.encode(label, return_tensors='pt')

    # test for only hellaswag
    kobest = lambda x: load_dataset('skt/kobest_v1', x)

    def eval_hellaswag():
        ds = kobest('hellaswag')['test']
        results = []
        for i, example in enumerate(tqdm(ds)):
            correct_label = example['label']
            wrong_labels = [0, 1, 2, 3]
            wrong_labels.remove(correct_label)
            random_template = random.choice(template_hellaswag)

            def make_hellaswag_prompt(template, pseudo_correct_label):
                correct_prompt = {k: v.format_map({
                    'context': example['context'],
                    'label': pseudo_correct_label,
                    'label_plusone': pseudo_correct_label + 1,
                    'ending_1': example['ending_1'],
                    'ending_2': example['ending_2'],
                    'ending_3': example['ending_3'],
                    'ending_4': example['ending_4'],
                    'correct_ending': example[f'ending_{pseudo_correct_label + 1}']
                }) for k, v in template.items()}
                return correct_prompt
            gold_inp = make_hellaswag_prompt(random_template, correct_label)
            wrong_inps = [make_hellaswag_prompt(random_template, wrong_label) for wrong_label in wrong_labels]

            res = []
            for inp in [gold_inp] + wrong_inps:
                prompt = prompter.generate_prompt(inp['instruction'])
                inp_tokens = tokenizer.encode(prompt, return_tensors='pt')
                cont_tokens = tokenizer.encode(inp['output'], return_tensors='pt')
                log_p = calc_log_p(model, inp_tokens, cont_tokens)
                res.append(log_p)
            if all(res[0] >= e for e in res[1:]):
                result = True
            else:
                result = False
            if i < 50:
                print(result)
            results.append(result)
        print("hellaswag score")
        print(sum(results) / len(results))

    print("hellaswag evaluation start...")
    eval_hellaswag()

    print("end")


if __name__ == '__main__':
    setproctitle("potatowook kullm eval")
    fire.Fire(main)


