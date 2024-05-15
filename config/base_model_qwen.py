max_new_tokens_factor_list = [4,8]
n_shots_inference_list = [0,2,4]
TRAIN_LAYER = 'en.layer1'

n_bit='NoQuant'
save_directory = f"data/qwen/7B_{n_bit}bit_base_newPaddToken"  # "data/gemma/test_data_processed"
BASE_MODEL_CHECKPOINT = "Qwen/Qwen1.5-7B-Chat"#"Qwen/Qwen1.5-14B-Chat"# "google/gemma-7b-it" # "mistralai/Mistral-7B-Instruct-v0.2"
instruction_on_response_format = 'Extract the entities contained in the text.\nReturn the result in a json format: [{"entity":"entity_name"}].'
simplest_prompt = False

import torch
torch_dtype = torch.bfloat16
batch_size = 24
