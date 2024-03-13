from dotenv import dotenv_values
from datasets import load_dataset
from utils.data_preprocessor import DataPreprocessor
from utils.test_data_processor import TestDataProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import base_model_qwen as base_model
import torch
from tqdm import tqdm


HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

max_new_tokens_factor_list = base_model.max_new_tokens_factor_list
n_shots_inference_list = base_model.n_shots_inference_list
layer = base_model.TRAIN_LAYER
language = layer.split('.')[0]
save_directory = base_model.save_directory 


dataset = load_dataset("ferrazzipietro/e3c-sentences", token=HF_TOKEN)
dataset = dataset[layer]
preprocessor = DataPreprocessor(model_checkpoint=base_model.BASE_MODEL_CHECKPOINT, 
                                tokenizer=base_model.BASE_MODEL_CHECKPOINT)

dataset = preprocessor.preprocess_data_one_layer(dataset, instruction_on_response_format=base_model.instruction_on_response_format,)
_, val_data, _ = preprocessor.split_layer_into_train_val_test_(dataset, layer)


load_in_4bit = False
load_in_4bit = False
if base_model.n_bit==4:
    load_in_4bit = True
if base_model.n_bit==8:
    load_in_8bit = True

if load_in_4bit or load_in_8bit: 
    bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_threshold= 6.0,
                llm_int8_skip_modules= ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"],
                )
    model = AutoModelForCausalLM.from_pretrained(
                base_model.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
                quantization_config = bnb_config,
                return_dict=True, 
                #torch_dtype=torch.float16,
                device_map= "auto",
                token=HF_TOKEN,
                #cache_dir='/data/disk1/share/pferrazzi/.cache'
    )
else:
    model = AutoModelForCausalLM.from_pretrained(base_model.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
                                                return_dict=True, device_map= "auto", token=HF_TOKEN)

tokenizer = AutoTokenizer.from_pretrained(base_model.BASE_MODEL_CHECKPOINT, add_eos_token=True, token=HF_TOKEN)
tokenizer.pad_token = '<pad>'
tokenizer.padding_side = "left"


for max_new_tokens_factor in max_new_tokens_factor_list:
    for n_shots_inference in n_shots_inference_list:
        
        # merged_model, tokenizer = load_mergedModel_tokenizer(adapters, base_model)
        postprocessor = TestDataProcessor(test_data=val_data, 
                                          preprocessor=preprocessor, 
                                          n_shots_inference=n_shots_inference, 
                                          language=language, 
                                          tokenizer=tokenizer)
        postprocessor.add_inference_prompt_column(simplest_prompt=False)
        postprocessor.add_ground_truth_column()
        print('TRY: ', f"{save_directory}/maxNewTokensFactor{max_new_tokens_factor}_nShotsInference{n_shots_inference}_BaseModel.csv")
        # try:
        postprocessor.add_responses_column(model=model, 
                                        tokenizer=tokenizer, 
                                        batch_size=3, 
                                        max_new_tokens_factor=max_new_tokens_factor)
        postprocessor.test_data.to_csv(f"{save_directory}/maxNewTokensFactor{max_new_tokens_factor}_nShotsInference{n_shots_inference}_BaseModel.csv", index=False)
        # except Exception as e: 
        #     print("ERROR IN PROCESSING: ", Exception)
