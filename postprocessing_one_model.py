from dotenv import dotenv_values
from datasets import load_dataset, Dataset
from utils.data_preprocessor import DataPreprocessor
from utils.evaluator import Evaluator
from config.finetuning import config
from utils.load_merged_model_tokenizer import load_mergedModel_tokenizer
from config import postprocessing
from utils.test_data_processor import TestDataProcessor
import pandas as pd
from log import enlayer1_3epochs_4bits__ft_params as models_params
from utils.generate_ft_adapters_list import generate_ft_adapters_list
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']


def run(test_data, preprocessor, n_shots_inference, language, tokenizer, adapters):
    max_new_tokens_factor_list = postprocessing.max_new_tokens_factor_list
    n_shots_inference_list = postprocessing.n_shots_inference_list
    layer = models_params.TRAIN_LAYER
    language = layer.split('.')[0]


    dataset = load_dataset("ferrazzipietro/e3c-sentences", token=HF_TOKEN)
    dataset = dataset[layer]
    preprocessor = DataPreprocessor()
    dataset = preprocessor.preprocess_data_one_layer(dataset)
    _, val_data, _ = preprocessor.split_layer_into_train_val_test_(dataset, layer)

    bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)

    base_model = AutoModelForCausalLM.from_pretrained(
                models_params.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
                quantization_config = bnb_config,
                return_dict=True,  #load_in_4bit=True, 
                torch_dtype=torch.float16,
                device_map= "auto")

    merged_model, tokenizer = load_mergedModel_tokenizer(adapters, base_model)

    postprocessor = TestDataProcessor(test_data=val_data.select(range(6)), 
                                              preprocessor=preprocessor, 
                                              n_shots_inference=n_shots_inference, 
                                              language=language, 
                                              tokenizer=tokenizer)
    postprocessor.add_inference_prompt_column()
    postprocessor.add_ground_truth_column()
    postprocessor.add_responses_column(model=merged_model, 
                                    tokenizer=tokenizer, 
                                    batch_size=3, 
                                    max_new_tokens_factor=max_new_tokens_factor)
    postprocessor.test_data.to_csv(f"data/test_data_processed/maxNewTokensFactor{max_new_tokens_factor}_nShotsInference{n_shots_inference}_{adapters.split('/')[1]}.csv", index=False)
            