from datasets import Dataset
from config.finetuning import preprocessing_params
import os
import random

class DataPreprocessor():


    def __init__(self) -> None:

        self.offset = None
        self.instruction_on_response_format = ''
        self.n_shots = None

        self.one_shot_example = """[INST] Extract the entities contained in the text and the offset, i.e. the position of that entity in the string. Extract only entities contained in the text.
{instruction_on_response_format} Text: <<<{example_query}>>> [/INST] {example_response}
"""
        self.one_shot_example_no_offset = """[INST] Extract the entities contained in the text. Extract only entities contained in the text.
{instruction_on_response_format} Text: <<<{example_query}>>> [/INST] {example_response}
"""

        self.prompt_template = """[INST] Extract the entities contained in the text and the offset, i.e. the position of that entity in the string. Extract only entities contained in the text.
{instruction_on_response_format} Text: <<{query}>>> [/INST]"""

        self.prompt_template_no_offset = """[INST] Extract the entities contained in the text. Extract only entities contained in the text.
{instruction_on_response_format} Text: <<{query}>>> [/INST]"""


    def _format_prompt(self, task: str, input: str, instruction_on_response_format:str, n_shots:int, offset: bool, tokenizer=None, output:str='', list_of_examples: [str]=[], list_of_responses:[str]=[]) -> str:
        """
        Format the input and output into a prompt for the finetuning

        Args:
            task: the task for which the prompt is generated, either 'finetuning' or 'inference'
            input: the input text
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            n_shots: the number of examples to provide as few shot prompting
            offset: whether to require the offset in the response
            tokenizer: the tokenizer to use
            output: the output text
            list_of_examples: the list of examples to provide as few shot prompting
            list_of_responses: the list of responses to provide as few shot prompting

        Returns:
            the formatted prompt
        """
        if task == 'finetuning':
            if n_shots > 0:
                raise ValueError("The numebr of shot in generating prompts for the finetuning must be 0")
            if tokenizer is None:
                raise ValueError("The tokenizer must be provided")
            if output == '':
                raise ValueError("The output must be provided when generating prompts for the finetuning")

        elif task == 'inference':
            if output != '':
                raise ValueError("The output must be an empty string when generating prompts for the inference")
        else:
            raise ValueError("The task must be either 'finetuning' or 'inference'")


        if len(list_of_examples) != len(list_of_responses):
            raise ValueError("The number of examples and responses must be the same")
        if n_shots != len(list_of_examples):
            raise ValueError("The number of examples and shots must be the same")
        if n_shots != len(list_of_responses):
            raise ValueError("The number of responses and shots must be the same")
        
        if offset:
            base_prompt = self.prompt_template.format(
                instruction_on_response_format=instruction_on_response_format, 
                query=input) 
            one_shot_example = self.one_shot_example
        else:
            base_prompt = self.prompt_template_no_offset.format(
                instruction_on_response_format=instruction_on_response_format, 
                query=input)
            one_shot_example = self.one_shot_example_no_offset
            
        prompt = ''
        for shot_example in range(n_shots):
            prompt += one_shot_example.format(
                instruction_on_response_format=instruction_on_response_format, 
                example_query=list_of_examples[shot_example], 
                example_response=list_of_responses[shot_example])
        
        bos_token = tokenizer.bos_token
        eos_token = ''
        if task == 'finetuning':
            eos_token = tokenizer.eos_token
        prompt = bos_token + prompt + base_prompt + output + eos_token
                            
        return prompt


    def _format_entities_in_response(self, entities_list: [dict], offset: bool) -> str:
        """
        Format the response into a string

        Args:
            entities_list: the list of entities to format
            offset: whether to require the offset in the response
            
        Returns:
            the formatted response
        """
        formatted_response = '['
        if offset:
            for entity in entities_list:
                formatted_response = formatted_response + '{"entity": "' + entity['text'] + f'", "offset": {entity["offsets"]}' + '}, '
        else:
            for entity in entities_list: 
                formatted_response = formatted_response + '{"entity": "' + entity['text'] + '"}, '
        formatted_response = formatted_response[:-2]
        formatted_response = formatted_response + '] '
        return formatted_response
    
    def _apply_to_one_example(self, example, task: str, instruction_on_response_format:str, n_shots:int, offset: bool, tokenizer=None, list_of_examples: [str]=[], list_of_responses:[str]=[]) -> dict:
        """
        Apply the data preprocessing to one example

        Args:
            example: the example (data row) to preprocess
            task: the task for which the prompt is generated, either 'finetuning' or 'inference'
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            n_shots: the number of examples to provide as few shot prompting
            offset: whether to require the offset in the response
            tokenizer: the tokenizer to use
            list_of_examples: the list of examples to provide as few shot prompting
            list_of_responses: the list of responses to provide as few shot prompting

        Returns:
            the preprocessed example
        """
        output = ''
        if task == 'finetuning':
            output = self._format_entities_in_response(entities_list=example['entities'], offset=offset)
        prompt = self._format_prompt(task, input=example['sentence'], instruction_on_response_format=instruction_on_response_format, n_shots=n_shots, offset=offset, tokenizer=tokenizer, output=output, list_of_examples=list_of_examples, list_of_responses=list_of_responses)
        example['prompt'] = prompt
        return example
    
    def apply(self, data: Dataset, task: str, instruction_on_response_format:str, n_shots:int, offset: bool, tokenizer=None, list_of_examples: [str]=[], list_of_responses:[str]=[], num_proc: int=1) -> Dataset:
        """
        Apply the data preprocessing to one split/layer if the dataset. It formats the prompt in the right shape, processing the entities.

        Args:
            data: the dataset to preprocess
            task: the task for which the prompt is generated, either 'finetuning' or 'inference'
            instruction_on_response_format: the instruction on the response format to be given to the model. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            n_shots: the number of examples to provide as few shot prompting   
            offset: whether to require the offset in the response  
            tokenizer: the tokenizer to use
            list_of_examples: the list of examples to provide as few shot prompting
            list_of_responses: the list of responses to provide as few shot prompting
            num_proc: the number of processes to use for the parallel processing

        Returns:
            the preprocessed split/layer
        """
        # , task, instruction_on_response_format, n_shots, offset, tokenizer, list_of_examples, list_of_responses
        data = data.map(lambda example:  self._apply_to_one_example(example, task, instruction_on_response_format, n_shots, offset, tokenizer, list_of_examples, list_of_responses), num_proc=num_proc) #batched=True)
        self.offset = offset
        self.instruction_on_response_format = instruction_on_response_format
        self.n_shots = n_shots
        return data
    
    def preprocess_data(self, hf_dataset: Dataset) -> Dataset:
        """
        Preprocess the dataset, applying to each split/layer the trasformations defined in self.apply()

        Args:
            hf_dataset: the dataset to preprocess

        Returns:
            the preprocessed dataset
        """
        splits = ['en.layer1', 'en.layer2', 'en.layer2.validation', 'en.layer3',
                'es.layer1', 'es.layer2', 'es.layer2.validation', 'es.layer3',
                'eu.layer1', 'eu.layer2', 'eu.layer2.validation', 'eu.layer3',
                'it.layer1', 'it.layer2', 'it.layer2.validation', 'it.layer3',
                'fr.layer1', 'fr.layer2', 'fr.layer2.validation', 'fr.layer3']
        for split_name in splits:
            hf_dataset[split_name] = self.apply(data=hf_dataset[split_name], 
                                                            task=preprocessing_params.task, 
                                                            instruction_on_response_format=preprocessing_params.instruction_on_response_format, 
                                                            n_shots=preprocessing_params.n_shots, 
                                                            offset=preprocessing_params.offset, 
                                                            tokenizer=preprocessing_params.tokenizer, 
                                                            list_of_examples=preprocessing_params.list_of_examples,
                                                            list_of_responses=preprocessing_params.list_of_responses)
        return hf_dataset
    
    def preprocess_data_one_layer(self, hf_dataset: Dataset) -> Dataset:
        """
        Preprocess one layer/split of the dataset the trasformations defined in self.apply()

        Args:
            hf_dataset: one layer/split of the dataset to preprocess

        Returns:
            the preprocessed dataset
        """
        hf_dataset = self.apply(data=hf_dataset, 
                                task=preprocessing_params.task, 
                                instruction_on_response_format=preprocessing_params.instruction_on_response_format, 
                                n_shots=preprocessing_params.n_shots, 
                                offset=preprocessing_params.offset, 
                                tokenizer=preprocessing_params.tokenizer, 
                                list_of_examples=preprocessing_params.list_of_examples,
                                list_of_responses=preprocessing_params.list_of_responses)
        return hf_dataset
    
    def split_layer_into_train_val_test_(self, dataset: Dataset, split_name: str, test_subset_of_validation: bool=False) -> (Dataset, Dataset):
        """
        Split the layer into train, validation and test sets, according to the split defined at https://github.com/hltfbk/E3C-Corpus/tree/main/documentation

        Args:
            dataset: the dataset to split. Must be a split of the original Hugging Face dataset
            split_name: the name of the layer
            test_subset_of_validation: wether the test set is a subset of the validation set. Set this to True if you want to use the test set as a way of checking on the training throw wandb
                                to mantain the diviosn it train-test of the original repository. Default is False.
        
        Returns:
            the train and test sets
        """
        mapping = {'en.layer1': 'train_labels_en.txt', 
                'es.layer1': 'train_labels_es.txt',
                'eu.layer1': 'train_labels_eu.txt',
                'it.layer1': 'train_labels_it.txt',
                'fr.layer1': 'train_labels_fr.txt',}
        labels_path = mapping[split_name]
        with open(os.path.join('data', labels_path), 'r') as file:
            file_content = file.read()
        labels = file_content.split(", ")
        labels = [label[1:-1] for label in labels]
        idxs_train = [idx for idx, x in enumerate(dataset['original_id']) if x in labels]
        idxs_val = [idx for idx, x in enumerate(dataset['original_id']) if x not in labels]
        random.seed(42)
        idxs_test = random.sample(idxs_val, int(len(idxs_val) * 0.2))
        train_data = dataset.select(idxs_train)
        test_data = dataset.select(idxs_test)
        if test_subset_of_validation:
            val_data = dataset.select(idxs_val)
        else:
            idxs_val = [idx for idx in idxs_val if idx not in idxs_test]
            val_data = dataset.select(idxs_val)

        if self.offset:
            prompt_template = self.prompt_template
        else:
            prompt_template = self.prompt_template_no_offset
        
        def remove_answer_from_prompt(example):
            prompt_no_answ = prompt_template.format(instruction_on_response_format=self.instruction_on_response_format, query=example['sentence'])
            example['prompt_with_answer'] = example['prompt']
            example['prompt'] = prompt_no_answ
            return example

        test_data = test_data.map(remove_answer_from_prompt, batched=False)

        return train_data, val_data, test_data
