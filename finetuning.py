# https://colab.research.google.com/github/adithya-s-k/LLM-Alchemy-Chamber/blob/main/LLMs/Mistral-7b/Mistral_Colab_Finetune_ipynb_Colab_Final.ipynb?source=post_page-----0f39647b20fe--------------------------------#scrollTo=acCr5AZ0831z

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, gradio, warnings
import torch
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import notebook_login
from dotenv import dotenv_values
from config import training_params, lora_params, model_loading_params, config

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
WANDB_KEY = dotenv_values(".env.base")['WANDB_KEY']
# Monitering the LLM
wandb.login(key = WANDB_KEY)

dataset = load_dataset(config.DATASET_CHEKPOINT) #download_mode="force_redownload"
dataset = dataset[config.TRAIN_LAYER]


print(dataset)

# Load base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit= model_loading_params.load_in_4bit,
    load_in_8bit = model_loading_params.load_in_8bit,

    bnb_4bit_quant_type= model_loading_params.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype= model_loading_params.bnb_4bit_compute_dtype,
    bnb_4bit_use_double_quant= model_loading_params.bnb_4bit_use_double_quant,

    llm_int8_threshold= model_loading_params.llm_int8_threshold,
    llm_int8_skip_modules= model_loading_params.llm_int8_skip_modules,
    llm_int8_has_fp16_weight= model_loading_params.llm_int8_has_fp16_weight
)

model = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL_CHECKPOINT,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1 # Tensor parallelism rank used during pretraining. This value is necessary to ensure exact reproducibility of the pretraining results
model.gradient_checkpointing_enable() # Activates gradient checkpointing for the current model.


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_CHECKPOINT, trust_remote_code=True, padding_side='left')
# tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
dataset = dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)

#Adding the adapters in the layers
"""
prepare_model_for_kbit_training wraps the entire protocol for preparing a model before running a training. 
        This includes:  1- Cast the layernorm in fp32 
                        2- making output embedding layer require gradient (needed as you are going to train (finetune) the model)
                        3- upcasting the model's head to fp32 for numerical stability
"""
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        r=lora_params.r,
        lora_alpha=lora_params.lora_alpha,
        lora_dropout=lora_params.lora_dropout,
        bias=lora_params.bias,
        task_type=lora_params.task_type,
        target_modules=lora_params.target_modules
        )
model = get_peft_model(model, peft_config)

#Hyperparamter
training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= training_params.num_train_epochs,
    per_device_train_batch_size= training_params.per_device_train_batch_size,
    gradient_accumulation_steps= training_params.gradient_accumulation_steps,
    optim=  training_params.optim,
    save_steps= training_params.save_steps,
    logging_steps= training_params.logging_steps,
    learning_rate= training_params.learning_rate,
    weight_decay= training_params.weight_decay,
    fp16= training_params.fp16,
    bf16= training_params.bf16,
    max_grad_norm= training_params.max_grad_norm,
    max_steps= training_params.max_steps,
    warmup_ratio= training_params.warmup_ratio,
    group_by_length= training_params.group_by_length,
    lr_scheduler_type= training_params.lr_scheduler_type,
    # remove_unused_columns=False
)


if config.train_on_subset:
    dataset = dataset.select(range(config.train_subset_size))


def formatting_prompts_func(example):
    return(example)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    #formatting_func=formatting_prompts_func,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=training_params.packing,
    max_seq_length= training_params.max_seq_length,
    dataset_text_field= training_params.dataset_text_field,
    data_collator= DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(dataset)
print(dataset['text'][0])
print(f"####### training_params.packing:{training_params.packing}\ntraining_params.max_seq_length={training_params.max_seq_length}\n training_params.dataset_text_field={training_params.dataset_text_field}")
# Monitering the LLM
run = wandb.init(project='Fine tuning', job_type="training", anonymous="allow")
trainer.train()
wandb.finish()

trainer.push_to_hub(config.FT_MODEL_CHECKPOINT, token = HF_TOKEN)