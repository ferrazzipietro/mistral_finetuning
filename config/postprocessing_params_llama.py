import torch
torch_dtype = torch.bfloat16
<<<<<<< HEAD
max_new_tokens_factor_list = [4,6,8] #,6,8] # [8]
n_shots_inference_list = [0] # [0,2,4] # [0] # 
save_directory = 'data/llama/7B_NoQuant_FT_cl_v2prompt/' # 'data/llama/7B_NoQuant_FT/'
batch_size = 48# 1
=======
max_new_tokens_factor_list = [4,6,8, 10] #,6,8] # [8]
n_shots_inference_list = [0] # [0,2,4] # [0] # 
save_directory = 'data/llama/7B_NoQuant_FT_cl_v2prompt_lowlora/' # 'data/llama/7B_NoQuant_FT/'
batch_size = 48# 1
temperature = 0.2
>>>>>>> 034d187f475f6f1e69f3d240c35d18b72dbf00ac
