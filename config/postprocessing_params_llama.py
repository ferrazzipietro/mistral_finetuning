import torch
torch_dtype = torch.bfloat16
max_new_tokens_factor_list = [2,4,8] # [8]
n_shots_inference_list = [0,2,4] # [0] # 
save_directory = 'data/llama/7B_8bit_FT/' # 'data/llama/7B_NoQuant_FT/'
batch_size = 12# 1
