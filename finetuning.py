from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, gradio, warnings
import torch
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import notebook_login
from dotenv import dotenv_values

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
dataset_checkpoint = dotenv_values(".env.base")['DATASET_CHEKPOINT']
base_model = "mistralai/Mistral-7B-v0.1"
new_model = dotenv_values(".env.base")['FT_MODEL_CHECKPOINT'] 

dataset = load_dataset(dataset_checkpoint)

# Load base model(Mistral 7B)
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= True,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

#Adding the adapters in the layers
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side='left')
# tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

def formatting_prompts_func(example):
    return(example)

#Hyperparamter
training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 2,
    per_device_train_batch_size= 8,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps= 1000,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant",
    report_to="wandb"
)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['it.layer1'].select(range(5)),
    #formatting_func=formatting_prompts_func,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="input",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()
trainer.push_to_hub(new_model, token = HF_TOKEN)