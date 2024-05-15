import os
from typing import List
from pathlib import Path

import fire
import setproctitle
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import concatenate_datasets
from pklue import get_mixture
from trl import DataCollatorForCompletionOnlyLM


def train(
    debug: bool = False,
    # model/data params
    base_model: str = "EleutherAI/polyglot-ko-1.3b",
    bf16: bool = False,
    output_dir: str = "./output",
    data_mixture: List[str] = None,
    max_examples: int = None,
    ds_config_file: str = None,
    custom_chat_template: str = None,
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
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training KULLM model with params:\n"
            f"{debug=}\n"
            f"{max_examples=}\n"
            f"{ds_config_file=}\n"
            f"base_model: {base_model}\n"
            f"{bf16=}\n"
            f"output_dir: {output_dir}\n"
            f"data_mixture: {data_mixture}\n"
            f"{custom_chat_template=}\n"
            f"per_device_train_batch_size: {per_device_train_batch_size}\n"
            f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"{use_wandb=}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    model = AutoModelForCausalLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, add_eos_token=True)
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id == tokenizer.pad_token_id:
        if tokenizer.unk_token_id is None:
            raise NotImplementedError("Unexpected situation. Assign pad_token_id to something. (but not eos_token_id)")
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if 'solar' in base_model.lower():
        tokenizer.model_max_length = 4096  # since solar default setting is ridiculous

    if custom_chat_template:
        with open(custom_chat_template, 'rt') as f:
            chat_template = f.read()
        tokenizer.chat_template = chat_template

    collator = DataCollatorForCompletionOnlyLM(
        response_template="[/INST]",
        instruction_template="[INST]",
        tokenizer=tokenizer
    )

    def make_inputs(item):
        chat = item['chat']
        messages = [{'role': e[0], 'content': e[1]} for e in chat]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, truncation=True)
        return {'input_ids': inputs}

    # will be removed, and truncating function will move to pKLUE.
    if not debug:
        # Truncating 'square' data
        if 'square_gpt4_sampled' in data_mixture:  # experimental feature: use small part of SQuARE dataset
            data_mixture.remove('square_gpt4_sampled')
            data = get_mixture(dataset_names=data_mixture, max_examples=max_examples, split='train')
            data_square = get_mixture(dataset_names=['square_gpt4_sampled'], max_examples=200, split='train')
            data = concatenate_datasets([data, data_square])
        else:
            data = get_mixture(dataset_names=data_mixture, max_examples=max_examples, split='train')
    else:
        # since aya_ko is small dataset, let's use this.
        data = get_mixture(dataset_names=['aya_ko'], max_examples=200, split='train')

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(make_inputs)
        val_data = train_val["test"].shuffle().map(make_inputs)
    else:
        train_data = data.shuffle().map(make_inputs)
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
            eval_steps=eval_steps if val_set_size > 0 else None,
            # save_steps=save_steps,
            output_dir=output_dir,
            load_best_model_at_end=True if val_set_size > 0 else False,
            report_to="wandb" if use_wandb else [],
            run_name=wandb_run_name if use_wandb else None,
            fp16=not bf16,
            bf16=bf16,
            gradient_checkpointing=True,
            deepspeed=ds_config_file,
        ),
        data_collator=collator
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # save modified tokenizer into checkpoint directory
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
