from datasets import Dataset
from utils.evaluator import Evaluator
from utils.output_cleaner import OutputCleaner
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

if __name__ == "__main__":
    from config import evaluation_params_all
    input_data_path = evaluation_params_all.input_data_dir_path
    output_data_path = evaluation_params_all.output_data_path
    words_level = evaluation_params_all.words_level
    evaluate(input_data_path, output_data_path, words_level, 
             similar_is_equal_threshold_list = evaluation_params_all.similar_is_equal_threshold_list,
             similarity_types = ['case', 'stop_words', 'subset', 'superset'],
             wrong_keys_to_entity = False,
             offset = False)
