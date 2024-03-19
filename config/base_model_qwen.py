max_new_tokens_factor_list = [4,8]
n_shots_inference_list = [0,1,2,3,4]
TRAIN_LAYER = 'en.layer1'

n_bit=4
save_directory = f"data/qwen/14B_{n_bit}bit_base"  # "data/gemma/test_data_processed"
BASE_MODEL_CHECKPOINT = "Qwen/Qwen1.5-14B-Chat"#"Qwen/Qwen1.5-14B-Chat"# "google/gemma-7b-it" # "mistralai/Mistral-7B-Instruct-v0.2"
instruction_on_response_format = 'Extract the entities contained in the text.\nReturn the result in a json format: [{"entity":"entity_name"}].'
simplest_prompt = False

batch_size = 30
