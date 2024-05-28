from dotenv import dotenv_values
from datasets import load_dataset, Dataset
from utils.data_preprocessor import Slovenian_preprocessor
from utils.test_data_processor import TestDataProcessor
from utils.generate_ft_adapters_list import generate_ft_adapters_list
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

from config import postprocessing_params_llama as postprocessing
from log import slo_llama7B_NoQuant as models_params
adapters_list = generate_ft_adapters_list("slo_llama7B_NoQuant", simplest_prompt=models_params.simplest_prompt)
print(adapters_list)
HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
LLAMA_TOKEN = dotenv_values(".env.base")['LLAMA_TOKEN']

max_new_tokens_factor_list = postprocessing.max_new_tokens_factor_list
n_shots_inference_list = postprocessing.n_shots_inference_list
layer = models_params.TRAIN_LAYER
language = 'slovenian'

tokenizer = AutoTokenizer.from_pretrained(models_params.BASE_MODEL_CHECKPOINT, add_eos_token=False,
                                         token=LLAMA_TOKEN)


val_data = pd.read_csv(models_params.slovenian_test_path, header=None, names=['word', 'label'])
preprocessor = Slovenian_preprocessor(val_data, models_params.BASE_MODEL_CHECKPOINT, tokenizer, token_llama=HF_TOKEN)
preprocessor.preprocess()
preprocessor.apply('', offset=False, simplest_prompt=False)
val_data = preprocessor.data
val_data = val_data.shuffle(seed=1234)  # Shuffle dataset here
val_data = val_data.map(lambda samples: tokenizer(samples[models_params.dataset_text_field]), batched=True)

for max_new_tokens_factor in max_new_tokens_factor_list:
    for n_shots_inference in n_shots_inference_list:
        for adapters in tqdm(adapters_list, desc="adapters_list"):

            print("PROCESSING:", adapters, "n_shots_inference:", n_shots_inference, "max_new_tokens_factor:", max_new_tokens_factor)
            if not models_params.quantization:
                print("NO QUANTIZATION")
                base_model = AutoModelForCausalLM.from_pretrained(
                    models_params.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
                    return_dict=True,  
                    torch_dtype=postprocessing.torch_dtype,
                    device_map= "auto",
                    token=LLAMA_TOKEN)    
            else:
                print("QUANTIZATION")
                load_in_8bit = not models_params.load_in_4bit[0]
                load_in_4bit = models_params.load_in_4bit[0]
                load_in_8bit = not load_in_4bit
                bnb_4bit_use_double_quant = models_params.bnb_4bit_use_double_quant
                bnb_4bit_quant_type = models_params.bnb_4bit_quant_type[0]
                bnb_4bit_compute_dtype = models_params.bnb_4bit_compute_dtype[0]
                llm_int8_threshold = models_params.llm_int8_threshold[0]
                # llm_int8_has_fp16_weight = models_params.llm_int8_has_fp16_weight AVOID IT AT INFERENCE TIME!
                # llm_int8_skip_modules = models_params.llm_int8_skip_modules AVOID IT AT INFERENCE TIME!

                bnb_config = BitsAndBytesConfig(
                            load_in_4bit=load_in_4bit,
                            load_in_8bit=load_in_8bit,
                            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                            bnb_4bit_quant_type=bnb_4bit_quant_type,
                            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                            llm_int8_threshold=llm_int8_threshold ,
                            # llm_int8_has_fp16_weight =True #,AVOID IT AT INFERENCE TIME!
                            # llm_int8_skip_modules=llm_int8_skip_modules AVOID IT AT INFERENCE TIME!
                            )
                base_model = AutoModelForCausalLM.from_pretrained(
                    models_params.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
                    quantization_config = bnb_config,
                    return_dict=True,  
                    device_map= "auto",
                    token=LLAMA_TOKEN)
            merged_model = PeftModel.from_pretrained(base_model, 
                                                     adapters, 
                                                     token=HF_TOKEN, 
                                                     device_map='auto',
                                                     is_trainable = False)
            tokenizer = AutoTokenizer.from_pretrained(models_params.BASE_MODEL_CHECKPOINT, 
                                                      add_eos_token=False,
                                                      token=LLAMA_TOKEN)
            tokenizer.pad_token = tokenizer.eos_token# "<pad>" #tokenizer.eos_token
            tokenizer.padding_side = "left"
#            tokenizer = AutoTokenizer.from_pretrained(models_params.BASE_MODEL_CHECKPOINT, add_eos_token=True, token=LLAMA_TOKEN)
#            tokenizer.add_special_tokens({"pad_token":"<pad>"})
#            merged_model.resize_token_embeddings(len(tokenizer))
#            print('tokenizer.pad_token_id:', tokenizer.pad_token_id)
#            merged_model.config.pad_token_id = tokenizer.pad_token_id

            postprocessor = TestDataProcessor(test_data=val_data, 
                                              preprocessor=preprocessor, 
                                              n_shots_inference=n_shots_inference, 
                                              language=language, 
                                              tokenizer=tokenizer)
            postprocessor.add_inference_prompt_column(simplest_prompt=False)

            # tmp = []
            # for example in postprocessor.test_data:
            #     tmp.append(example)
            # import pandas as pd
            # tmp = pd.DataFrame(tmp)
            # tmp = tmp.iloc[tmp['inference_prompt'].str.len().argsort()]
            # postprocessor.test_data = Dataset.from_pandas(tmp)

            postprocessor.add_ground_truth_column()
            #try:
            postprocessor.add_responses_column(model=merged_model, 
                                            tokenizer=tokenizer, 
                                            batch_size=postprocessing.batch_size, 
                                            max_new_tokens_factor=max_new_tokens_factor)
            postprocessor.test_data.to_csv(f"{postprocessing.save_directory}maxNewTokensFactor{max_new_tokens_factor}_nShotsInference{n_shots_inference}_{adapters.split('/')[1]}.csv", index=False)
            # except RuntimeError as e:
                # print("ERROR IN PROCESSING: ", e, adapters)
                # print(e.message)
            del merged_model
            if models_params.quantization: del base_model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()



