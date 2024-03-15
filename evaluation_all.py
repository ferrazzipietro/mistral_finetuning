from evaluation import evaluate
import os
from config import evaluation_params_all

WORDS_LEVEL = evaluation_params_all.words_level
similar_is_equal_threshold_list = evaluation_params_all.similar_is_equal_threshold_list
similarity_types = evaluation_params_all.similarity_types
wrong_keys_to_entity = evaluation_params_all.wrong_keys_to_entity
offset = evaluation_params_all.offset

if __name__ == "__main__":
    data_dir = 'data/'
    # directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    # print('directories:', directories)
    errors = []
    for directory in ['llama', 'qwen', 'mistral']:
        sub_directories = [d for d in os.listdir(os.path.join(data_dir, directory)) if os.path.isdir(os.path.join(data_dir, directory, d))]
        for sub_dir in sub_directories:
            if sub_dir.startswith('old'):
                continue
            input_dir_path = os.path.join(data_dir, directory, sub_dir)
            output_file_name = directory + '_' + sub_dir + f'_wordsLevel{WORDS_LEVEL}' + '_evaluation.csv'
            print('input_dir_path', input_dir_path)
            output_file_path = os.path.join(data_dir, 'evaluation_results', output_file_name)
            #print('output_file_path', output_file_path)
            new_errors = evaluate(input_data_path=input_dir_path, 
                            output_file_path = output_file_path, 
                            words_level=WORDS_LEVEL,
                            similar_is_equal_threshold_list=similar_is_equal_threshold_list,
                            similarity_types = similarity_types,
                            wrong_keys_to_entity = wrong_keys_to_entity,
                            offset = offset)
            errors.extend(new_errors)


    print('errors:', errors)