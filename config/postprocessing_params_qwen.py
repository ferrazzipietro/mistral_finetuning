import torch

max_new_tokens_factor_list = [2,4,8]
n_shots_inference_list = [2,4] #[0, 2, 4]
save_directory = 'data/qwen/7B_NoQuant_FT/'
batch_size = 12
torch_dtype=torch.float16
