# https://colab.research.google.com/github/adithya-s-k/LLM-Alchemy-Chamber/blob/main/LLMs/Mistral-7b/Mistral_Colab_Finetune_ipynb_Colab_Final.ipynb?source=post_page-----0f39647b20fe--------------------------------#scrollTo=acCr5AZ0831z

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from trl import SFTTrainer
from dotenv import dotenv_values
from config import training_params, lora_params, model_loading_params, config
import wandb
from utils.data_preprocessor import DataPreprocessor
from utils.wandb_callback import LLMSampleCB


HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
WANDB_KEY = dotenv_values(".env.base")['WANDB_KEY']
FT_MODEL_CHECKPOINT = config.FT_MODEL_CHECKPOINT #Name of the model you will be pushing to huggingface model hub

# Monitering the LLM
wandb.login(key = WANDB_KEY)
run = wandb.init(project='Fine tuning en.layer1', job_type="training", anonymous="allow")

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

model_id = config.BASE_MODEL_CHECKPOINT

model = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL_CHECKPOINT,
    quantization_config=bnb_config,
    device_map="auto"
)
model.gradient_checkpointing_enable() # Activates gradient checkpointing for the current model.
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
#Adding the adapters in the layers
"""
prepare_model_for_kbit_training wraps the entire protocol for preparing a model before running a training. 
        This includes:  1- Cast the layernorm in fp32 
                        2- making output embedding layer require gradient (needed as you are going to train (finetune) the model)
                        3- upcasting the model's head to fp32 for numerical stability
"""
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_CHECKPOINT, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

preprocessor = DataPreprocessor()
dataset = load_dataset(config.DATASET_CHEKPOINT) #download_mode="force_redownload"
dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
dataset = preprocessor.preprocess_data(dataset)
dataset = dataset.map(lambda samples: tokenizer(samples[training_params.dataset_text_field]), batched=True)
dataset = dataset[config.TRAIN_LAYER]
train_data, val_data, test_data = preprocessor.split_layer_into_train_val_test_(dataset, config.TRAIN_LAYER)

def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)
modules = find_all_linear_names(model)

lora_config = LoraConfig(
        r=lora_params.r,
        lora_alpha=lora_params.lora_alpha,
        lora_dropout=lora_params.lora_dropout,
        bias=lora_params.bias,
        task_type=lora_params.task_type,
        target_modules=lora_params.target_modules
        )
model = get_peft_model(model, lora_config)

torch.cuda.empty_cache()

#Hyperparamter
training_arguments = TrainingArguments(
    output_dir= "./training_output",
    push_to_hub=True,
    hub_model_id=config.FT_MODEL_CHECKPOINT,
    hub_token=HF_TOKEN,
    hub_private_repo=True,
    num_train_epochs= training_params.num_train_epochs,
    per_device_train_batch_size= training_params.per_device_train_batch_size,
    per_device_eval_batch_size= training_params.per_device_train_batch_size/2,
    gradient_accumulation_steps= training_params.gradient_accumulation_steps,
    optim=  training_params.optim,
    save_steps= training_params.save_steps,
    logging_strategy=training_params.logging_strategy,
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
    report_to="wandb",
    #lr_scheduler_type="cosine",
    #warmup_ratio = 0.1,

    # logging strategies 
    # remove_unused_columns=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    dataset_text_field=training_params.dataset_text_field,
    peft_config=lora_config,
    args=training_arguments,
    max_seq_length = training_params.max_seq_length,
    # Currently (01/'24) Packing is not supported with Instruction Masking (data_collator argument is not supported with packing=True)
    # just packing without instruction masking gives good results already
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), # see here: https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy#(optional)-preprocessing:-masking-instructions-by-using-the-datacollatorforcompletiononlylm
    packing=False,# True would create a ConstantLengthDataset so it can iterate over the dataset on fixed-length sequences
    neftune_noise_alpha=5,
)

wandb_callback = LLMSampleCB(trainer, test_data, num_samples=10, max_new_tokens=256)
trainer.add_callback(wandb_callback)

trainer.train()

trainer.model.save_pretrained(f"{config.BASE_MODEL_CHECKPOINT.split('/')[1]}_prova") # save locally
trainer.model.push_to_hub(config.ADAPTERS_CHECKPOINT, token=HF_TOKEN)


# del model, trainer
# torch.cuda.empty_cache()

# # Reload the base model
# base_model_reload = AutoModelForCausalLM.from_pretrained(
#     config.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
#     return_dict=True,torch_dtype=torch.bfloat16,
#     device_map= "auto")

# merged_model = PeftModel.from_pretrained(base_model_reload, f"{config.ADAPTERS_CHECKPOINT.split('/')[1]}_prova", token=HF_TOKEN)
# merged_model = merged_model.merge_and_unload()


# # Reload tokenizer
# tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_CHECKPOINT, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"


# print(type(merged_model))
# # Save the merged model
# # merged_model.save_pretrained(f"{config.FT_MODEL_CHECKPOINT}_local",safe_serialization=True)
# # tokenizer.save_pretrained(f"{config.FT_MODEL_CHECKPOINT}_local")
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# # Push the model and tokenizer to the Hugging Face Model Hub
# merged_model.push_to_hub(config.FT_MODEL_CHECKPOINT, use_temp_dir=False, token=HF_TOKEN )
# tokenizer.push_to_hub(config.FT_MODEL_CHECKPOINT, use_temp_dir=False, token=HF_TOKEN )


wandb.finish()