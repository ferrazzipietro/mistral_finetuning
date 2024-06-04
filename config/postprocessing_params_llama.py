import torch
torch_dtype = torch.bfloat16
max_new_tokens_factor_list = [4,6,8, 10] #,6,8] # [8]
n_shots_inference_list = [0] # [0,2,4] # [0] # 
save_directory = 'data/llama/7B_NoQuant_FT_cl_1epoch/' # 'data/llama/7B_NoQuant_FT/'
batch_size = 48# 1
temperature = 1
