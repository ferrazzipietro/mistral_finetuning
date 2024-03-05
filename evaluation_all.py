from evaluation import evaluate
import os
from config import postprocessing

WORDS_LEVEL = postprocessing.words_level
similar_is_equal_threshold_list = postprocessing.similar_is_equal_threshold_list
similarity_types = postprocessing.similarity_types
wrong_keys_to_entity = postprocessing.wrong_keys_to_entity
offset = postprocessing.offset



if __name__ == "__main__":
    data_dir = 'data/'
    directories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print('directories:', directories)
    errors = []
    for directory in directories:
        if 'evaluation_results' in directory:
            continue 
        sub_directories = [d for d in os.listdir(os.path.join(data_dir, directory)) if os.path.isdir(os.path.join(data_dir, directory, d))]
        for sub_dir in sub_directories:
            input_dir_path = os.path.join(data_dir, directory, sub_dir)
            output_file_name = directory + '_' + sub_dir + f'_wordsLevel{WORDS_LEVEL}' + '_evaluation.csv'
            print('input_dir_path', input_dir_path)
            output_file_path = os.path.join(data_dir, 'evaluation_results', output_file_name)
            #print('output_file_path', output_file_path)
            try:
                evaluate(input_data_path=input_dir_path, 
                        output_file_path = output_file_path, 
                        words_level=WORDS_LEVEL,
                        similar_is_equal_threshold_list=similar_is_equal_threshold_list,
                        similarity_types = similarity_types,
                        wrong_keys_to_entity = wrong_keys_to_entity,
                        offset = offset)
            except Exception as e:
                errors.append({input_dir_path: e})

    print('errors:', errors)