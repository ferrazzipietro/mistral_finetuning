from datasets import Dataset

class DataPreprocessor():

    def __init__(self) -> None:

        self.one_shot_example = """[INST] Extract the entities contained in the text and the offset, i.e. the position of that entity in the string. Extract only entities contained in the text.
{instruction_on_response_format}
Text: <<<{example_query}>>> [/INST]
{example_response}
"""
        self.one_shot_example_no_offset = """[INST] Extract the entities contained in the text. Extract only entities contained in the text.
{instruction_on_response_format}
Text: <<<{example_query}>>> [/INST]
{example_response}
"""

        self.prompt_template = """[INST] Extract the entities contained in the text and the offset, i.e. the position of that entity in the string. Extract only entities contained in the text.
{instruction_on_response_format}
Text: <<{query}>>> [/INST]
"""

        self.prompt_template_no_offset = """
<s>[INST] Extract the entities contained in the text. Extract only entities contained in the text.
{instruction_on_response_format}
Text: <<{query}>>> [/INST]
"""


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
            response: the response to format
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
        Apply the data preprocessing to the dataset

        Args:
            data: the dataset to preprocess

        Returns:
            the preprocessed dataset
        """
        # , task, instruction_on_response_format, n_shots, offset, tokenizer, list_of_examples, list_of_responses
        data = data.map(lambda example:  self._apply_to_one_example(example, task, instruction_on_response_format, n_shots, offset, tokenizer, list_of_examples, list_of_responses), num_proc=num_proc) #batched=True)
        return data
