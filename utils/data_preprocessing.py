from config import preprocessing_params
from .data_preprocessor import DataPreprocessor

def preprocess_data(hf_dataset):
    data_preprocessor = DataPreprocessor()
    splits = ['en.layer1', 'en.layer2', 'en.layer2.validation', 'en.layer3',
            'es.layer1', 'es.layer2', 'es.layer2.validation', 'es.layer3',
            'eu.layer1', 'eu.layer2', 'eu.layer2.validation', 'eu.layer3',
            'it.layer1', 'it.layer2', 'it.layer2.validation', 'it.layer3',
            'fr.layer1', 'fr.layer2', 'fr.layer2.validation', 'fr.layer3']
    for split_name in splits:
        hf_dataset[split_name] = data_preprocessor.apply(data=hf_dataset[split_name], 
                                                        task=preprocessing_params.task, 
                                                        instruction_on_response_format=preprocessing_params.instruction_on_response_format, 
                                                        n_shots=preprocessing_params.n_shots, 
                                                        offset=preprocessing_params.offset, 
                                                        tokenizer=preprocessing_params.tokenizer, 
                                                        list_of_examples=preprocessing_params.list_of_examples,
                                                        list_of_responses=preprocessing_params.list_of_responses)
    return hf_dataset
