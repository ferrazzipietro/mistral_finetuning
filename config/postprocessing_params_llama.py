import torch
torch_dtype = torch.float16
max_new_tokens_factor_list = [4,8]
n_shots_inference_list = [0, 2, 4] # [0,2,4] #
save_directory = 'data/llama/7B_8bit_FT/'
batch_size = 36