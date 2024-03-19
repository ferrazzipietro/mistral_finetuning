# from evaluation import evaluate
import os
from config import evaluation_params_all


from datasets import Dataset
from utils.evaluator import Evaluator
# from utils.output_cleaner import OutputCleaner
import glob
import pandas as pd

def evaluate(input_data_path:str, 
             output_file_path:str, 
             words_level:bool,
             similar_is_equal_threshold_list,
             similarity_types:list = ['case', 'stop_words', 'subset', 'superset'],
             wrong_keys_to_entity=False,
             offset=False) -> list:
    """
    Evaluate the model on the test data contained in the input_data_path directory
    
    Args:
    input_data_path: The path to the test data directory
    output_data_path: The path to the output file
    words_level: Whether to evaluate the model at the word level
    similar_is_equal_threshold_list: The list of similarity thresholds to evaluate the model
    similarity_types: The list of similarity types to evaluate the model. Optional, default is ['case', 'stop_words', 'subset', 'superset']
    wrong_keys_to_entity: Whether to consider wrong keys as entities. Optional, default is False
    offset: Whether to consider the offset of the entities. Optional, default is False
    
    Returns:
    lsit: A list of files where errors occurred during the evaluation 
    """
    csv_files = glob.glob(input_data_path + '/*.csv') 
    evaluation_results = pd.DataFrame(columns=['file', 'similar_is_equal', 'similar_is_equal_threshold', 'f1_score', 'precision', 'recall'])
    errors = []
    for file in csv_files:
        print("FILE: " , file)
        eval_data = Dataset.from_csv(file) 
        try:
            output_cleaner = OutputCleaner(verbose=False)
            cleaned_data = output_cleaner.apply_cleaning(eval_data, wrong_keys_to_entity=wrong_keys_to_entity)
            for similar_is_equal_threshold in similar_is_equal_threshold_list:
                evaluator = Evaluator(data=cleaned_data, offset=offset, output_cleaner=output_cleaner)
                evaluator.generate_evaluation_table(similar_is_equal_threshold=similar_is_equal_threshold,
                                                    words_level=words_level, 
                                                    similarity_types=similarity_types)
                evaluation_results.loc[len(evaluation_results)] = {'file': file, 'similar_is_equal_threshold': similar_is_equal_threshold, 'f1_score': evaluator.evaluation_table['f1'], 'precision': evaluator.evaluation_table['precision'], 'recall': evaluator.evaluation_table['recall']}
        except Exception as e:
            errors.extend({file: e})
    evaluation_results.to_csv(output_file_path, index=False)
    return errors





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