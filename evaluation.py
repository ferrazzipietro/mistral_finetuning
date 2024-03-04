from datasets import Dataset
from config import postprocessing
from utils.evaluator import Evaluator
from utils.output_cleaner import OutputCleaner
import glob
import pandas as pd
#adapters_list = generate_ft_adapters_list("enlayer1_3epochs_4bits__ft_params")



similar_is_equal_list = postprocessing.similar_is_equal_list
similar_is_equal_threshold_list = postprocessing.similar_is_equal_threshold_list


def evaluate(input_data_path:str, output_data_path:str, words_level) -> pd.DataFrame:
    """
    Evaluate the model on the test data
    
    Args:
    input_data_path: The path to the test data directory
    output_data_path: The path to the output file
    
    Returns:
    evaluation_results: The evaluation results
    """
    csv_files = glob.glob(input_data_path + '*.csv') 
    evaluation_results = pd.DataFrame(columns=['file', 'similar_is_equal', 'similar_is_equal_threshold', 'f1_score', 'precision', 'recall'])

    for file in csv_files:
        print("FILE: " , file)
        eval_data = Dataset.from_csv(file) 
        output_cleaner = OutputCleaner()
        cleaned_data = output_cleaner.apply_cleaning(eval_data, wrong_keys_to_entity=False)
        for similar_is_equal in similar_is_equal_list:
            if similar_is_equal:
                for similar_is_equal_threshold in similar_is_equal_threshold_list:
                    evaluator = Evaluator(data=cleaned_data, offset=False, output_cleaner=output_cleaner)
                    evaluator.generate_evaluation_table(similar_is_equal=similar_is_equal, 
                                                        similar_is_equal_threshold=100,
                                                        words_level=words_level, 
                                                        similarity_types=['case', 'stop_words', 'subset', 'superset'])#,'leveshtein'])
                    evaluation_results.loc[len(evaluation_results)] = {'file': file, 'similar_is_equal': similar_is_equal, 'similar_is_equal_threshold': similar_is_equal_threshold, 'f1_score': evaluator.evaluation_table['f1'], 'precision': evaluator.evaluation_table['precision'], 'recall': evaluator.evaluation_table['recall']}
            elif not similar_is_equal:
                evaluator = Evaluator(data=cleaned_data, offset=False, output_cleaner=output_cleaner)
                evaluator.generate_evaluation_table(similar_is_equal=similar_is_equal, 
                                                    similar_is_equal_threshold=100,
                                                    words_level=words_level,
                                                    similarity_types=['case', 'stop_words', 'subset', 'superset'])
                evaluation_results.loc[len(evaluation_results)] = {'file': file, 'similar_is_equal': similar_is_equal, 'similar_is_equal_threshold': similar_is_equal_threshold, 'f1_score': evaluator.evaluation_table['f1'], 'precision': evaluator.evaluation_table['precision'], 'recall': evaluator.evaluation_table['recall']}
        
    evaluation_results.to_csv(output_data_path, index=False)
    return evaluation_results

if __name__ == "__main__":
    input_data_path = postprocessing.input_data_dir_path
    output_data_path = postprocessing.output_data_path
    words_level = postprocessing.words_level
    evaluate(input_data_path, output_data_path, words_level)
