import pandas as pd
import string
from utils.slovenian_preprocessor import Slovenian_preprocessor
import pandas as pd
import string
from datasets import Dataset
from transformers import AutoTokenizer


class Slovenian_preprocessor():


    def __init__(self, data, model_checkpoint:str, tokenizer: AutoTokenizer, token_llama:str='') -> None:

        self.data = data
        self.offset = None
        self.instruction_on_response_format = ''
        self.n_shots = None
        #self.model_type = model_checkpoint.split('/')[1].lower().split('-')[0]
        self.model_type = 'qwen' if model_checkpoint.split('/')[0] == 'Qwen' else model_checkpoint.split('/')[1].lower().split('-')[0]
        if self.model_type == 'meta': self.model_type = 'llama3'
        # if self.model_type == 'zefiro':
        #     self.model_type  = 'mistral'
        if self.model_type not in ['mistral', 'llama', 'llama3', 'gemma', 'qwen', 'zefiro', 'phi', 'minerva']:
            raise ValueError("The model type must be either 'mistral', 'llama', 'llama3', 'gemma', 'zefiro', 'qwen', 'minerva' or 'phi'")

        print('MODEL TYPE:', self.model_type)
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, token = token_llama)
        else:
            self.tokenizer = tokenizer
        
        self.special_tokens_instruction_dict = {'mistral': {'user_start':'[INST]',
                                                            'user_end':'[/INST]',
                                                            'model_start':'',
                                                            'model_end':''},
                                                'llama': {'user_start':'[INST]',
                                                          'user_end':'[/INST]',
                                                          'model_start':'',
                                                          'model_end':''},
                                                'llama3': {'user_start':'<|start_header_id|>user<|end_header_id|>\n\n',
                                                          'user_end':'<|eot_id|>',
                                                          'model_start':'<|start_header_id|>assistant<|end_header_id|>\n\n',
                                                          'model_end':''},
                                                'gemma': {'user_start':'<start_of_turn>user',
                                                          'user_end':'<end_of_turn>',
                                                          'model_start':'<start_of_turn>model',
                                                          'model_end':'<end_of_turn>'},
                                                'qwen': {'user_start':'<|im_start|>user',
                                                          'user_end':'<|im_end|>',
                                                          'model_start':'<|im_start|>assistant',
                                                          'model_end':'<|im_end|>'},
                                                'zefiro': {'user_start':'<|user|>',
                                                           'user_end':'',# 'user_end':'</s>',
                                                           'model_start':'<|assistant|>',
                                                           'model_end':''},# 'model_end':'</s>'},
                                                'phi': {'user_start':'<|user|>',
                                                           'user_end':'<|end|>\n',
                                                           'model_start':'<|assistant|>',
                                                           'model_end':''},
                                                'minerva': {'user_start':'',
                                                            'user_end':'',
                                                           'model_start':'',
                                                           'model_end':''}}
        self.special_tokens_instruction = self.special_tokens_instruction_dict[self.model_type]

        self.one_shot_example = """{user_start} {instruction_on_response_format} <<<{example_query}>>> {user_end}{model_start} {example_response} {model_end}
"""
        self.one_shot_example_no_offset = """{user_start} {instruction_on_response_format} <<<{example_query}>>> {user_end}{model_start} {example_response} {model_end}
"""

        self.prompt_template = """{user_start} {instruction_on_response_format} <<{query}>>> {user_end}{model_start}"""

        self.prompt_template_no_offset = """{user_start} {instruction_on_response_format} <<{query}>>> {user_end}{model_start}"""

    def _base_prompt_input(self, input: str, instruction_on_response_format:str) -> str:
        """
        Format the input into a base prompt for the finetuning

        Args:
            input: the input text
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"

        Returns:
            the formatted base prompt
        """
        base_prompt = self.prompt_template_no_offset.format(
            instruction_on_response_format=instruction_on_response_format, 
            query=input,
            user_start=self.special_tokens_instruction['user_start'],
            user_end=self.special_tokens_instruction['user_end'],
            model_start=self.special_tokens_instruction['model_start'],
            model_end=self.special_tokens_instruction['model_end'])
            
        return base_prompt

    def _simplest_base_prompt_input(self, input: str) -> str:
        """
        Format the input and output into a prompt for the finetuning, in the simplest way possible, containing only the sentence and the response

        Args:
            input: the input text
            output: the output text

        Returns:
            the formatted prompt
        """
        base_prompt = self.special_tokens_instruction['user_start'] + input + self.special_tokens_instruction['user_end'] + self.special_tokens_instruction['model_start']
        return base_prompt

    def _format_prompt(self, input: str, instruction_on_response_format:str, simplest_prompt: bool, output:str='') -> str:
        """
        Format the input and output into a prompt for the finetuning

        Args:
            input: the input text
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            offset: whether to require the offset in the response
            output: the output text

        Returns:
            the formatted prompt
        """
        if output == '':
            raise ValueError("The output must be provided when generating prompts for the finetuning")
        
        if simplest_prompt:
            prompt_input = self._simplest_base_prompt_input(input)
        else:
            prompt_input = self._base_prompt_input(input, instruction_on_response_format)
        
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        if self.model_type == 'qwen':
            bos_token = ''
            eos_token = ''
        # print(bos_token, prompt_input, output, self.special_tokens_instruction['model_end'], eos_token)
        prompt = bos_token + prompt_input + output + self.special_tokens_instruction['model_end'] + eos_token
                            
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
                formatted_response = formatted_response + '{"entity": "' + entity['entity'] + f'", "offset": {entity["offsets"]}' + '}, '
        else:
            for entity in entities_list: 
                formatted_response = formatted_response + '{"entity": "' + entity['entity'] + '"}, '
        formatted_response = formatted_response[:-2]
        formatted_response = formatted_response + '] '
        return formatted_response
    
    def _apply_to_one_example(self, example, offset: bool, simplest_prompt: bool, instruction_on_response_format:str) -> dict:
        """
        Apply the data preprocessing to one example

        Args:
            example: the example (data row) to preprocess
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            offset: whether to require the offset in the response
            simplest_prompt: whether to generate the prompt or just concatenate the sentence and the response

        Returns:
            the preprocessed example
        """
        output = self._format_entities_in_response(entities_list=example['entities'], offset=offset)
        prompt = self._format_prompt(input=example['text'], 
                                     simplest_prompt=simplest_prompt,
                                     instruction_on_response_format=instruction_on_response_format,
                                     output=output)
        example['prompt'] = prompt
        return example
    
    def apply(self, instruction_on_response_format:str, offset: bool,  simplest_prompt:bool, num_proc: int=1): # -> Dataset:
        """
        Apply the data preprocessing to one split/layer if the dataset. It formats the prompt in the right shape, processing the entities.

        Args:
            data: the dataset to preprocess
            instruction_on_response_format: the instruction on the response format to be given to the model. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            n_shots: the number of examples to provide as few shot prompting   
            offset: whether to require the offset in the response  
            num_proc: the number of processes to use for the parallel processing

        Returns:
            the preprocessed split/layer
        """
        self.data = self.data.map(lambda example:  self._apply_to_one_example(example=example, 
                                                                    simplest_prompt=simplest_prompt,
                                                                    instruction_on_response_format = instruction_on_response_format, 
                                                                    offset = offset), 
                        num_proc=num_proc) #batched=True)
        self.offset = offset
        self.instruction_on_response_format = instruction_on_response_format
        self.simplest_prompt = simplest_prompt
        # return data

    
    def preprocess(self):
        self.data['label'] = self.data['label'].astype(str)
        text = ''
        overall_entities = []
        entities = []
        current_entity = ''
        prev_word=''
        for _, row in self.data.iterrows():
            # print(f"word: {row['word']}, label: {row['label']}, prev_word: {prev_word}")
            if prev_word == '.' and len(text.split())>5:
                if current_entity:
                    entities.append({'entity': current_entity})
                    current_entity = ''
                # print(f"ENTRO NELLA COND")
                overall_entities.append({'text': text, 'entities': entities})
                text = ''
                entities = []
            word, label = row['word'], row['label']
            if text!='' or (word.strip() in string.punctuation) or (prev_word.strip() in string.punctuation):
                space = ' '
            else:
                space = ''
            text = text + space + word
            if label != 'O':
                if label.startswith('B-'):
                    if current_entity:
                        entities.append({'entity': current_entity})
                    current_entity = word
                elif label.startswith('I-') and current_entity:
                    current_entity += ' ' + word
            prev_word = word
        data_df = pd.DataFrame(columns=['text', 'entities'])
        for el in overall_entities:
            el['text'] = el['text'].strip() 
            data_df = pd.concat([data_df, pd.DataFrame([el])], ignore_index=True)
        self.data = Dataset.from_pandas(data_df, split='train')
