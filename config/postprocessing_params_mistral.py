import torch
max_new_tokens_factor_list = [4,8]
n_shots_inference_list = [0,2,4] #[2, 4]
save_directory = 'data/mistral/8bit_FT/'
batch_size = 24
torch_dtype=torch.float16

