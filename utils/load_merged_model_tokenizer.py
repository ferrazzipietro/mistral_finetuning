from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
from dotenv import dotenv_values

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

def load_mergedModel_tokenizer(adapters_checkpoint, model_checkpoint):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model_reload = AutoModelForCausalLM.from_pretrained(
        model_checkpoint, low_cpu_mem_usage=True,
        quantization_config = bnb_config,
        return_dict=True,  load_in_4bit=True, #torch_dtype=torch.float16,
        device_map= "auto")
    merged_model = PeftModel.from_pretrained(base_model_reload, adapters_checkpoint, token=HF_TOKEN, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return merged_model, tokenizer

def load_tokenizer(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer