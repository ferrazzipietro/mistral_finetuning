max_new_tokens_factor_list = [2,4,8]# [2,4,8]
n_shots_inference_list = [0,1,2,3,4]
TRAIN_LAYER = 'en.layer1'

n_bit=None
save_directory = "data/mistral/NoQuant_base_bfloat16" # f"data/mistral/{n_bit}bit_base" 
BASE_MODEL_CHECKPOINT = "mistralai/Mistral-7B-Instruct-v0.2"# "google/gemma-7b-it" # "mistralai/Mistral-7B-Instruct-v0.2"
instruction_on_response_format = 'Extract the entities contained in the text.\nReturn the result in a json format: [{"entity":"entity_name"}].'
simplest_prompt = False

batch_size = 36

import torch
dtype = torch.bfloat16