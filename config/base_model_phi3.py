import torch

max_new_tokens_factor_list = [4,8]
n_shots_inference_list = [0,1,2,3,4]
TRAIN_LAYER = 'en.layer1'

n_bit=4 # 'NoQuant'
save_directory = f"data/phi3/{n_bit}bit_base" # "data/gemma/test_data_processed"
BASE_MODEL_CHECKPOINT = "microsoft/Phi-3-mini-4k-instruct"# "google/gemma-7b-it" # "mistralai/Mistral-7B-Instruct-v0.2"

instruction_on_response_format = 'Extract the entities contained in the text.\nReturn the result in a json format: [{"entity":"entity_name"}].'
simplest_prompt = False

torch_dtype = torch.bfloat16
batch_size = 48
