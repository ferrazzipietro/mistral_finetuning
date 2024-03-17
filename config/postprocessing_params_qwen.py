max_new_tokens_factor_list = [2,4,8]
n_shots_inference_list = [2,4] #[2, 4]
save_directory = 'data/qwen/14B_4bit_FT/'
batch_size = 12

# EVALUATION
similar_is_equal_list=[True, False]
similar_is_equal_threshold_list=[100]#[95, 90, 85, 80, 75, 70, 65, 60, 100]
words_level = True
input_data_dir_path = 'data/qwen/4bit'
output_data_path = 'data/evaluation_results/mistral_4bit.csv'


similarity_types = ['case', 'stop_words', 'subset', 'superset', 'levenshtein']
wrong_keys_to_entity = False
offset = False

