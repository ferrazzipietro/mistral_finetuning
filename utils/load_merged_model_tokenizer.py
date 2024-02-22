from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
from dotenv import dotenv_values

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

def load_mergedModel_tokenizer(adapters_checkpoint, base_model, task:str = "inference", device_map:str="auto", llama_key:str=""):

    if isinstance(base_model, str):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
            
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model, low_cpu_mem_usage=True,
            quantization_config = bnb_config,
            return_dict=True,  load_in_4bit=True, #torch_dtype=torch.float16,
            device_map= device_map,
            token=llama_key)
        
    model_checkpoint = base_model.config._name_or_path
        
    merged_model = PeftModel.from_pretrained(base_model, adapters_checkpoint, token=HF_TOKEN, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    if task == "inference":
        tokenizer.padding_side = "left"
    if task == "training":
        tokenizer.padding_side = "right"
    return merged_model, tokenizer

def load_tokenizer(model_checkpoint, task:str = "inference"):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    if task == "inference":
        tokenizer.padding_side = "left"
    if task == "training":
        tokenizer.padding_side = "right"
    return tokenizer