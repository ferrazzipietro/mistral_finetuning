# https://colab.research.google.com/github/adithya-s-k/LLM-Alchemy-Chamber/blob/main/LLMs/Mistral-7b/Mistral_Colab_Finetune_ipynb_Colab_Final.ipynb?source=post_page-----0f39647b20fe--------------------------------#scrollTo=acCr5AZ0831z

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from trl import SFTTrainer
from dotenv import dotenv_values
from config.finetuning import training_params, lora_params, model_loading_params, config
import wandb
from utils.data_preprocessor import DataPreprocessor
import datetime


HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
WANDB_KEY = dotenv_values(".env.base")['WANDB_KEY']
FT_MODEL_CHECKPOINT = config.FT_MODEL_CHECKPOINT #Name of the model you will be pushing to huggingface model hub

# Monitering the LLM
wandb.login(key = WANDB_KEY)
run = wandb.init(project=config.ADAPTERS_CHECKPOINT.split('/')[1], job_type="training", anonymous="allow",
                 name=config.TRAIN_LAYER+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 config={'model': config.BASE_MODEL_CHECKPOINT, 
                         'dataset': config.DATASET_CHEKPOINT, 
                         'layer': config.TRAIN_LAYER,
                         'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
"""Args:
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            This flag is used to enable 8-bit quantization with LLM.int8().
        llm_int8_threshold (`float`, *optional*, defaults to 6.0):
            This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix
            Multiplication for Transformers at Scale` paper: https://arxiv.org/abs/2208.07339 Any hidden states value
            that is above this threshold will be considered an outlier and the operation on those values will be done
            in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
            there are some exceptional systematic outliers that are very differently distributed for large models.
            These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
            magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
            but a lower threshold might be needed for more unstable models (small models, fine-tuning).
        llm_int8_skip_modules (`List[str]`, *optional*):
            An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
            Jukebox that has several heads in different places and not necessarily at the last position. For example
            for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
        llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`):
            This flag is used for advanced use cases and users that are aware of this feature. If you want to split
            your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
            this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
            operations will not be run on CPU.
        llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`):
            This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
            have to be converted back and forth for the backward pass."""
bnb_config = BitsAndBytesConfig(
    load_in_4bit= False,# model_loading_params.load_in_4bit,
    load_in_8bit = True,#  model_loading_params.load_in_8bit,

    # bnb_4bit_quant_type= model_loading_params.bnb_4bit_quant_type[0],
    # bnb_4bit_compute_dtype= model_loading_params.bnb_4bit_compute_dtype[0],
    # bnb_4bit_use_double_quant= model_loading_params.bnb_4bit_use_double_quant,

    llm_int8_threshold= 6.0,# model_loading_params.llm_int8_threshold,
    llm_int8_skip_modules= ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"],# model_loading_params.llm_int8_skip_modules,
    # llm_int8_has_fp16_weight= True# model_loading_params.llm_int8_has_fp16_weight
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
dataset = dataset[config.TRAIN_LAYER]
dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
dataset = preprocessor.preprocess_data_one_layer(dataset)
dataset = dataset.map(lambda samples: tokenizer(samples[training_params.dataset_text_field]), batched=True)
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
        r=16,# lora_params.r,
        lora_alpha=32,#lora_params.lora_alpha,
        lora_dropout=0.05, #lora_params.lora_dropout,
        bias=lora_params.bias,
        task_type=lora_params.task_type,
        target_modules=lora_params.target_modules
        )
model = get_peft_model(model, lora_config)

print('model.is_loaded_in_8bit: ', model.is_loaded_in_8bit)
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
    gradient_accumulation_steps=2, # training_params.gradient_accumulation_steps,
    optim=  training_params.optim,
    save_steps= training_params.save_steps,
    logging_strategy=training_params.logging_strategy,
    logging_steps= training_params.logging_steps,
    learning_rate=2e-4, #training_params.learning_rate,
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
 #   packing=False,# True would create a ConstantLengthDataset so it can iterate over the dataset on fixed-length sequences
 #   neftune_noise_alpha=5,
)
# from wandb.keras import WandbCallback
# # wandb_callback = LLMSampleCB(trainer, test_data, num_samples=10, max_new_tokens=256)
from utils.wandb_callback import PrinterCallback
# trainer.add_callback(PrinterCallback)
from utils.wandb_callback import WandbPredictionProgressCallback
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=test_data,
    num_samples=10,
    freq=2,
)

# Add the callback to the trainer
trainer.add_callback(progress_callback)

with torch.autocast("cuda"):
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