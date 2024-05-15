import torch
max_new_tokens_factor_list = [4,8]
n_shots_inference_list = [0, 4]# ,2,4] #[2, 4]
save_directory = 'data/mistral/4bit_FT/'
batch_size = 18
torch_dtype=torch.bfloat16

