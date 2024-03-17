from dotenv import dotenv_values
from datasets import load_dataset, Dataset
from utils.data_preprocessor import DataPreprocessor
from utils.evaluator import Evaluator
from config.finetuning import config
from config import postprocessing_params_mistral as postprocessing
from utils.test_data_processor import TestDataProcessor
import pandas as pd
from utils.generate_ft_adapters_list import generate_ft_adapters_list
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
from peft import PeftModel
from tqdm import tqdm

adapters = "ferrazzipietro/Mistral-7B-Instruct-v0.2__adapters_en.layer1_4_torch.bfloat16_16_32_0.05_2_0.0002"
print(adapters)
BASE_MODEL_CHECKPOINT = 'mistralai/Mistral-7B-Instruct-v0.2'

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

max_new_tokens_factor = 6
n_shots_inference = 0
layer = 'en.layer1'
language = layer.split('.')[0]
quantization  = False


dataset = load_dataset("ferrazzipietro/e3c-sentences", token=HF_TOKEN)
dataset = dataset[layer]
preprocessor = DataPreprocessor(model_checkpoint=BASE_MODEL_CHECKPOINT, 
                                tokenizer =BASE_MODEL_CHECKPOINT)
dataset = preprocessor.preprocess_data_one_layer(dataset,
                                                 simplest_prompt=False,
                                                 instruction_on_response_format='Extract the entities contained in the text. Extract only entities contained in the text.\nReturn the result in a json format: [{"entity":"entity_name"}].')
_, val_data, _ = preprocessor.split_layer_into_train_val_test_(dataset, layer)

if not quantization:
    print("NO QUANTIZATION")
    merged_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
        return_dict=True,  
        torch_dtype=postprocessing.torch_dtype,
        device_map= "auto")    
# else:
    # print("QUANTIZATION")
    # load_in_8bit = not models_params.load_in_4bit[0]
    # bnb_config = BitsAndBytesConfig(
    #             load_in_4bit = models_params.load_in_4bit[0],
    #             load_in_8bit = load_in_8bit,
    #             bnb_4bit_use_double_quant = models_params.bnb_4bit_use_double_quant,
    #             bnb_4bit_quant_type = models_params.bnb_4bit_quant_type[0],
    #             bnb_4bit_compute_dtype = models_params.bnb_4bit_compute_dtype[0],
    #             llm_int8_threshold = models_params.llm_int8_threshold[0],
    #             llm_int8_has_fp16_weight = models_params.llm_int8_has_fp16_weight,
    #             llm_int8_skip_modules = models_params.llm_int8_skip_modules
    #             )
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     models_params.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
    #     quantization_config = bnb_config,
    #     return_dict=True,  
    #     #torch_dtype=torch.float16,
    #     device_map= "auto")
    # merged_model = PeftModel.from_pretrained(base_model, adapters, token=HF_TOKEN, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
# merged_model, tokenizer = load_mergedModel_tokenizer(adapters, base_model)
postprocessor = TestDataProcessor(test_data=val_data, 
                                  preprocessor=preprocessor, 
                                  n_shots_inference=n_shots_inference, 
                                  language=language, 
                                  tokenizer=tokenizer)
postprocessor.add_inference_prompt_column(simplest_prompt=False)
postprocessor.add_ground_truth_column()
try:
        postprocessor.add_responses_column(model=merged_model, 
                                        tokenizer=tokenizer, 
                                        batch_size=34, 
                                        max_new_tokens_factor=max_new_tokens_factor)
        postprocessor.test_data.to_csv(f"data/TMP_maxNewTokensFactor{max_new_tokens_factor}_nShotsInference{n_shots_inference}_{adapters.split('/')[1]}.csv", index=False)
except Exception as e:
    print("ERROR IN PROCESSING: ", Exception, adapters)
# del merged_model
# if models_params.quantization: 
#     del base_model
# del tokenizer
# gc.collect()
# torch.cuda.empty_cache()