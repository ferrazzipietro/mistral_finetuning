from utils.data_preprocessor import DataPreprocessor
from config import preprocessing_params
from datasets import Dataset
from tqdm import tqdm
import json
import re

class DataPostprocessor():
    def __init__(self, test_data: Dataset, preprocessor:DataPreprocessor, n_shots_inference:int, language:str, tokenizer) -> None:
        self.test_data = test_data
        self.preprocessor = preprocessor
        self.language = language
        self.tokenizer = tokenizer
        self.few_shots_dict = {'eng':{'questions':['We present a case of a 32-year-old woman with a history of gradual enlargement of the anterior neck.',
                                                   'Patient information: a 9-month-old boy presented to the emergency room with a 3-day history of refusal to bear weight on the right lower extremity and febrile peaks of up to 38.5°C for 24 hours.'],
                                        'responses':['[{"entity": "present"}, {"entity": "history"}, {"entity": "enlargement"}]',
                                                     '[{"entity": "presented"}, {"entity": "refusal"}, {"entity": "bear"}, {"entity": "peaks"}]'],
                                        'responses_offset': ['[{"entity": "present", "offset": [3, 10]}, {"entity": "history", "offset": [48, 55]}, {"entity": "enlargement", "offset": [67, 78]}]',
                                                             '[{"entity": "presented", "offset": [39, 48]}, {"entity": "refusal", "offset": [95, 102]}, {"entity": "bear", "offset": [106, 110]}, {"entity": "peaks", "offset": [159, 164]}]']
                                    },
                                'ita':{'questions':[],
                                       'responses':[],
                                       'responses_offset':[]}
                                }
        if len(self.few_shots_dict[self.language]['questions']) < n_shots_inference:
            raise ValueError(f'The number of shots for the inference prompt is greater than the number of examples available.')
        if len(self.few_shots_dict[self.language]['responses']) < n_shots_inference:
            raise ValueError(f'The number of shots for the inference prompt is greater than the number of responses available.')
        self.n_shots_inference = n_shots_inference
    
    def _extract_ground_truth(self, prompt:str) -> str:
        out = prompt.split('[/INST]', 1)
        return {'ground_truth': out[1][0:-4].strip()}
        
    def _extract_inference_prompt(self, sentence:str) -> str:
        if self.preprocessor.offset:
            few_shots_responses = self.few_shots_dict[self.language]['responses_offset']
        else:
            few_shots_responses = self.few_shots_dict[self.language]['responses']
        inference_prompt = self.preprocessor._format_prompt(task='inference', 
                                                        input=sentence, 
                                                        instruction_on_response_format=self.preprocessor.instruction_on_response_format,
                                                        offset=self.preprocessor.offset,
                                                        tokenizer=self.tokenizer,
                                                        output='',
                                                        n_shots=self.n_shots_inference,
                                                        list_of_examples=self.few_shots_dict[self.language]['questions'][0:self.n_shots_inference],
                                                        list_of_responses=few_shots_responses)
        return {'inference_prompt': inference_prompt}
    
    def add_inference_prompt_column(self) -> None:
        """
        Add the inferencePrompt and groundTruth columns to the test_data dataframe.
        """
        self.test_data = self.test_data.map(lambda x: self._extract_inference_prompt(x['sentence']))
    
    def add_ground_truth_column(self) -> None:
        """
        Add the groundTruth column to the test_data dataframe.
        """
        self.test_data = self.test_data.map(lambda x: self._extract_ground_truth(x['prompt']))

    def _generate_model_response(self, examples, model, tokenizer, max_new_tokens_factor:float=4) -> str:
        device = "cuda"
        tokenizer.padding_side = "left"
        input_sentences = examples['sentence']
        prompts = examples['inference_prompt']
        input_sentences_tokenized = tokenizer(input_sentences, return_tensors="pt", padding=True)
        max_new_tokens = int(len(max(input_sentences_tokenized, key=len)) * max_new_tokens_factor)

        encodeds = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True)
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(**model_inputs, do_sample=True, max_new_tokens=max_new_tokens,  pad_token_id=tokenizer.eos_token_id) # max_new_tokens=max_new_tokens,
        decoded = tokenizer.batch_decode(generated_ids)
        decoded = [self._postprocess_model_output(i) for i in decoded]
        return (decoded)
                
    def add_responses_column(self, model, tokenizer, batch_size:int) -> None:
        """
        Adds a column with the response of the model to the actual query.
        
        params:
        model: the model to use to generate the response
        tokenizer: the tokenizer to use to generate the response
        batch_size: the batch size to use to process the examples. Increasing this makes it faster but requires more GPU. Default is 8.
        """
        responses_col = []
        total_rows = len(self.test_data)
        indexes = [i for i in range(len(self.test_data)) if i % batch_size == 0]
        max_index = self.test_data.shape[0]


        with tqdm(total=total_rows, desc="generating responses") as pbar:
            for i, idx in enumerate(indexes[:-1]):
                indici = list(range(idx, indexes[i+1]))
                tmp = self._generate_model_response(self.test_data.select(indici), model, tokenizer)
                responses_col.extend(tmp)
                pbar.update(batch_size)
            indici = list(range(indexes[i+1], max_index))
            tmp = self._generate_model_response(self.test_data.select(indici), model, tokenizer)
            responses_col.extend(tmp)
            pbar.update(batch_size)

        self.test_data = self.test_data.add_column('model_responses', responses_col)
    
    def _postprocess_model_output(self, model_output: str) -> str:
        """
        Postprocess the model output to return a json like formatted string that can be used to compute the F1 score.

        Args:
        model_output (str): the model output as it is returned by the model. The processing of the output is done in the function

        return:
        str: the model response, i.e. the model output without the instruction

        """
        def has_unclosed_square_brackets(s):
            count = 0
            for char in s:
                if char == '[':
                    count += 1
                elif char == ']':
                    count -= 1
                    if count < 0:
                        return True
            return count > 0
        
        model_output = model_output.split('[/INST]')[-1].strip()
        if self._assess_model_output(model_output):
            return model_output
        
        tmp = re.findall(r'\[\{(.+?)\}\]', model_output)
        if len(tmp) != 0:
            tmp = '[{' + tmp[0] + '}]'
            if self._assess_model_output(tmp):
                return tmp
        if has_unclosed_square_brackets(model_output):
            last_bracket_index = model_output.rfind('},') # find the last complete entity
            model_output = model_output[:last_bracket_index+1] + ']' 
            return model_output 
  
        
    def _assess_model_output(self, model_response: str) -> bool:
        """
        Check if the model output is in the right format. If not, return False.
        
        Args:
        model_output (str): the postprocessed model output after beeing passed to _postprocess_model_output()

        return:
        bool: True if the format is correct, False otherwise
        """
        good_format = True
        try :
            res = json.loads(model_response)
            print( res)
        except:
            good_format = False
        return good_format

    