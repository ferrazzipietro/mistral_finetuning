import json
import re

class OutputCleaner():
    def __init__(self) -> None:
        pass
  
    def _remove_space_from_dict_keys(self, model_ouput_list: list) -> list:
        """
        Remove the spaces from the keys of a dictionary. E.g., [{"entity ": "value"}] -> [{"entity": "value"}]

        Args:
        model_ouput_list (dict): the list of dictionaries to be cleaned

        return:
        list: the cleaned list of dicts
        """
        out = []
        for dict in model_ouput_list:
            out.append({k.replace(' ', ''): v for k, v in dict.items()})
        return out
    
    def _drop_duplicates(self, model_response: list) -> str:
        """
        Drop the duplicates from a list. This is useful when the model output contains the same entity multiple times.

        Args:
        model_response (str): the model response with no duplicates
        """
        try :
            return list({v['entity']:v for v in model_response}.values())
        except Exception as error:
            model_response = self._remove_space_from_dict_keys(model_response)
            return list({v['entity']:v for v in model_response}.values())
        
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
            # print( res)
        except:
            good_format = False
        return good_format
    
    def _clean_model_output(self, example: dict) -> str:
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
        
        def has_unopen_square_brackets(s):
            count = 0
            for char in s:
                if char == '[':
                    count -= 1
                elif char == ']':
                    count += 1
                    if count > 0:
                        return True
            return count > 0
        
        def is_list_of_lists(string):
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list) and all(isinstance(item, list) for item in tmp):
                    return True
            return False
        
        def is_list_of_strings(string):
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list) and all(isinstance(item, str) for item in tmp):
                    return True
            return False

        model_output = example['model_responses']

        if model_output is None:
            return {'model_output':'[{"entity":""}]'}
        
        if is_list_of_lists(model_output):
            tmp = json.loads(model_output)
            tmp = str(tmp[0])
            return {'model_output':tmp}

        if is_list_of_strings(model_output):
            tmp = json.loads(model_output)
            tmp = [{"entity":el} for el in tmp]
            tmp = str(tmp)
            # print('TMP: ', tmp)
            # raise Exception
            return {'model_output': tmp}

        
        if self._assess_model_output(model_output):
            return {'model_output':model_output}
        
        if has_unopen_square_brackets(model_output):
            last_bracket_index = model_output.rfind('],') # keep the closed list
            model_output = '[' + model_output[:last_bracket_index+1] 
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


        if model_output.strip()[0] == '{':
            tmp = '[' + model_output + ']'
            if self._assess_model_output(tmp):
                return {'model_output':tmp}
            else:
                last_bracket_index = model_output.rfind('},') # find the last complete entity
                model_output = '[' + model_output[:last_bracket_index+1] + ']'
                return {'model_output':model_output}
            
        if model_output.strip().startswith('[['):
            tmp = model_output[1:]
            if self._assess_model_output(tmp):
                return {'model_output':tmp}
        print('THIS IS A BROKEN ROW: ', model_output)

        return {'model_output':model_output}

    
    def apply_cleaning(self, data) -> None:
        """
        Apply the cleaning to the model output and return the cleaned response in a new cloumn called 'model_output

        Args:
        model_output (str): the model output as it is returned by the model. The processing of the output is done in the function

        return:
        str: the model response, i.e. the model output without the instruction
        """
        return data.map(lambda x: self._clean_model_output(x))