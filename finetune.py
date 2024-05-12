import copy
import os
import random
import sys
from typing import List
from pathlib import Path

import fire
import setproctitle
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk, concatenate_datasets
from pklue import get_mixture


def train(
    debug: bool = False,
    # model/data params
    base_model: str = "EleutherAI/polyglot-ko-1.3b",
    bf16: bool = False,
    output_dir: str = "./lora-alpaca",
    data_mixture: List[str] = None,
    max_examples: int = None,
    vram_available: str = None,
    add_kodata: bool = False,
    apply_chat_template: bool = False,
    # training hyperparams
    per_device_train_batch_size: int = 0,
    gradient_accumulation_steps: int = 0,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,
    val_set_size: int = 2000,
    warmup_ratio: float = 0.0,
    warmup_steps: int = 0,
    logging_steps: int = 1,
    eval_steps: int = 200,
    save_steps: int = 200,
    lr_scheduler_type: str = 'cosine',
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    # wandb params
    use_wandb: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "kullm",  # The prompt template to use, will default to alpaca.
):
    setproctitle.setproctitle(f"potatowook {Path(output_dir).name}")
    print(type(add_eos_token))
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training KULLM model with params:\n"
            f"{debug=}\n"
            f"{max_examples=}\n"
            f"{vram_available=}\n"
            f"base_model: {base_model}\n"
            f"{bf16=}\n"
            f"output_dir: {output_dir}\n"
            f"data_mixture: {data_mixture}\n"
            f"add_kodata: {add_kodata}\n"
            f"{apply_chat_template=}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
            f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"{use_wandb=}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    if bf16:
        fp16 = False
    else:
        fp16 = True

    # Check if parameter passed or if set within environ
    # print(f"{wandb_project=}")
    # use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    # # Only overwrite environ if wandb param passed
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    # print(f"{use_wandb=}")
    # use_wandb = False

    model = AutoModelForCausalLM.from_pretrained(base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    if 'mistral' in base_model or 'upstage/SOLAR-10.7B-v1.0' in base_model:
        print("no pad_token defined in tokenizer... Setting pad_token equal to eos_token...")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if apply_chat_template:
        with open('templates/kullm3_chat_template.txt', 'rt') as f:
            kullm3_chat_template = f.read()
        tokenizer.chat_template = kullm3_chat_template

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,  # Add when kullm3 development, since apply_chat_template inserts bos manually
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    # TODO need to refactor
    def generate_and_tokenize_prompt(data_point, random_chat_template=False):
        instruction = data_point['instruction']
        if random_chat_template:
            assert data_point['input'] == ""
            instruction = make_instruction_with_random_template(instruction)

        data_input = data_point['input'] if 'input' in data_point.keys() else None

        if apply_chat_template:
            # for kullm3 development
            instruction = tokenizer.apply_chat_template([{
                'role': 'user',
                'content': instruction
            }], tokenize=False)
            full_prompt = instruction + data_point['output']
        else:
            # legacy code (before kullm3)
            full_prompt = prompter.generate_prompt(
                instruction,
                # data_point["input"],
                data_input,
                data_point["output"],
            )
        # eos_token을 일단은 무작위로 주는데, 이게 맞나 싶다. 벤치마크를 위해서는 맞을지도?
        # 다시 보니까, eos_token은 문장 맨 마지막에만 주어지는 거니까 항상 True로 줘도 돼 보임.
        # random_eos_token = random.choice([True, False])
        tokenized_full_prompt = tokenize(full_prompt, add_eos_token=add_eos_token)
        if not train_on_inputs:
            if apply_chat_template:
                tokenized_user_prompt = tokenize(instruction, add_eos_token=add_eos_token)
            else:
                # user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
                user_prompt = prompter.generate_prompt(instruction, data_input)
                tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            # if add_eos_token:
            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if vram_available == "82GB":
        ds_config_file = "ds_config/experimenting_a100.json"
    elif vram_available == "48GB":
        ds_config_file = "ds_config/experimenting_a6000_nooffload.json"
    else:
        raise NotImplementedError

    if not debug:
        # Truncating KLUE dataset if exists, since it is too big compared to other datasets.
        if 'klue' in data_mixture:
            print("KLUE dataset usage detected! Truncating KLUE...")
            data_without_klue = copy.deepcopy(data_mixture)
            data_without_klue.remove('klue')
            data1 = get_mixture(dataset_names=data_without_klue, max_examples=max_examples, split='train')
            data2 = get_mixture(dataset_names=['klue'], max_examples=200, split='train')
            data = concatenate_datasets([data1, data2])
        else:
            if 'kullm3_square_gpt4_sampled' in data_mixture:  # experimental feature: use small part of SQuARE dataset
                data_mixture.remove('kullm3_square_gpt4_sampled')
                data = get_mixture(dataset_names=data_mixture, max_examples=max_examples, split='train')
                data_square = get_mixture(dataset_names=['kullm3_square_gpt4_sampled'], max_examples=200, split='train')
                data = concatenate_datasets([data, data_square])
            else:
                data = get_mixture(dataset_names=data_mixture, max_examples=max_examples, split='train')

        # Added feature: mix other data (KOpen-platypus, OpenOrca-KO)
        if add_kodata:
            print('\033[95m' + "Mixing Additional Korean Data..." + '\033[0m')
            platypus_ko = load_platypus_ko()
            openorca_ko = load_openorca_ko()
            data = concatenate_datasets([data, platypus_ko, openorca_ko])
    else:
        data = get_mixture(dataset_names=['kullm3_aya'], max_examples=200, split='train')

    # truncate 'klue' dataset to have up to 200 items for each subset

    print("---")
    print(data)
    print(f"{val_set_size}")
    print("---")

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data.shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            # warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=logging_steps,
            # optim="adamw_torch",  # since we use DS optim?
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="epoch",
            # eval_steps=200 if val_set_size > 0 else None,
            # eval_steps=eval_steps,
            # save_steps=save_steps,
            output_dir=output_dir,
            # save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            # ddp_find_unused_parameters=False if ddp else None,
            # ddp_find_unused_parameters=True,
            report_to="wandb" if use_wandb else [],
            run_name=wandb_run_name if use_wandb else None,
            fp16=fp16,
            bf16=bf16,
            # max_grad_norm=1.0,  # cutting edge issue, https://github.com/huggingface/transformers/pull/29212
            gradient_checkpointing=True,
            deepspeed=ds_config_file,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # save
    tokenizer.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
