from config.finetuning import preprocessing_params
from datasets import Dataset
from tqdm import tqdm
import json
import re

class OutputCleaner():
    def __init__(self, test_data: Dataset) -> None:
        self.test_data = test_data

    def _clean_model_output(self, model_output: str) -> str:
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
        
        if self._assess_model_output(model_output):
            return {'model_output':model_output}
        
        tmp = re.findall(r'\[\{(.+?)\}\]', model_output)
        if len(tmp) != 0:
            tmp = '[{' + tmp[0] + '}]'
            if self._assess_model_output(tmp):
                return {'model_output':tmp}
        if has_unclosed_square_brackets(model_output):
            last_bracket_index = model_output.rfind('},') # find the last complete entity
            model_output = model_output[:last_bracket_index+1] + ']' 
            return {'model_output':model_output} 
        return {'model_output':model_output}
  
        
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
  
    def apply_cleaning(self, model_output: str) -> str:
        """
        Apply the cleaning to the model output and return the cleaned response.

        Args:
        model_output (str): the model output as it is returned by the model. The processing of the output is done in the function

        return:
        str: the model response, i.e. the model output without the instruction
        """
        return self._clean_model_output(model_output)