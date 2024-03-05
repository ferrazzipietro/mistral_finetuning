import json
import re
from typing import Tuple
from typing import List

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

            
    def _remove_json_special_chars(self, string):
        chars = ['\xa0']
        for char in chars:
            string = string.replace(char, ' ')
        return string
    
    def _clean_ground_truth(self, example: dict) -> dict:
        ground_truth = example['ground_truth']
        ground_truth = self._remove_json_special_chars(ground_truth)
        if ground_truth == ']':
            ground_truth = '[' + ground_truth
        return({'ground_truth': ground_truth})

    def _clean_model_output(self, example: dict,  wrong_keys_to_entity:bool, latest_version:bool=True) -> dict:
        """
        Postprocess the model output to return a json like formatted string that can be used to compute the F1 score.

        Args:
        model_output (str): the model output as it is returned by the model. The processing of the output is done in the function
        wrong_keys_to_entity (bool): if True, the function also extracts the dictionaries with keys different from 'entity', converting the keys into 'entity'. If not, all keys that are not 'entity' are dropped

        return:
        dict: the model response

        """
        def has_unclosed_square_brackets(s:str)  -> bool:
            count = 0
            for char in s:
                if char == '[':
                    count += 1
                elif char == ']':
                    count -= 1
                    if count < 0:
                        return True
            return count > 0
        
        def has_unopen_square_brackets(s:str)  -> bool:
            count = 0
            for char in s:
                if char == '[':
                    count -= 1
                elif char == ']':
                    count += 1
                    if count > 0:
                        return True
            return count > 0
        
        def is_empty_list(string:str)  -> bool:
            if string=='[]':
                return True
            return False
        
        def is_list_of_lists(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list) and all(isinstance(item, list) for item in tmp):
                    return True
            return False
        
        def is_list_of_dicts(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list) and all(isinstance(item, dict) for item in tmp):
                    return True
            return False
        
        def is_list_of_lists_and_dict(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                found_dict = False
                found_list = False
                for element in tmp:
                    if isinstance(element, list):
                        found_list = True
                    elif isinstance(element, dict):
                        found_dict = True
                    if found_list and found_dict:
                        return True
            return False
        
        def is_list_of_strings(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list) and all(isinstance(item, str) for item in tmp):
                    return True
            return False

        def is_list_of_dicts_and_strings(string:str)  -> bool:
            if self._assess_model_output(string):
                #print('ASSESSED')
                tmp = json.loads(string)
                found_dict = False
                found_string = False
                for element in tmp:
                    if isinstance(element, str):
                        found_string = True
                    elif isinstance(element, dict):
                        found_dict = True
                    if found_string and found_dict:
                        return True
            return False
        
        def is_string(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, str):
                    return True
            return False
        
        def is_numeric(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, (int, float)):
                    return True
            return False
        
        def are_entities_extracted_as_dict_keys_instead_of_values(string:str, example:dict) -> bool:
            if is_list_of_dicts(string):
                tmp = json.loads(string)
                keys = [key for item in tmp for key in item.keys()]
                if 'entity' not in keys:
                    if all(entity in example['sentence'] for entity in keys):
                        return True
            return False
        
        
        
        def convert_wrong_keys_into_entity(string:str) -> List[str]:
            if is_list_of_dicts(string):
                tmp = json.loads(string)
                tmp = [str({"entity":v}) for el in tmp for v in el.values()]
                return tmp
            else:
                return []


        def only_dicts_with_key_entity(string:str, wrong_keys_to_entity:bool) -> Tuple[bool, str]:
            """
            Extract only the dictionaries with the key 'entity' in the list of dictionaries in the string
            
            Args:
            string (str): the string to be cleaned
            wrong_keys_to_entity (bool): if True, the function also extracts the dictionaries with keys different from 'entity', converting the keys into 'entity'
            """
            els_between_curly = re.findall(r'\{(.+?)\}', string)
            clean = [el for el in els_between_curly if el.startswith('"entity"') or el.startswith("'entity'")]
            clean = ['{' + el + '}' for el in clean]
            dirty = []
            if wrong_keys_to_entity:
                dirty = [el for el in els_between_curly if (not el.startswith('"entity"')) and (not el.startswith("'entity'"))]
                dirty = ['{' + el + '}' for el in dirty]
                dirty = '[' + ', '.join(dirty) + ']'
                cleaned_dirty = convert_wrong_keys_into_entity(dirty)
                out = '[' + ', '.join(clean) + ', '.join(cleaned_dirty) +  ']'
            else:
                out = '[' + ', '.join(clean) + ']'
            out = out.replace("{\'", "{\"").replace("\'}", "\"}").replace("\'ent", "\"ent").replace("ty\'", "ty\"").replace(" \'", " \"")
            operations_performed = False
            if len(clean) != len(els_between_curly):
                operations_performed = True
            if is_empty_list(out):
                return operations_performed, '[{"entity":""}]'
            return operations_performed, str(out)
        

        model_output = example['model_responses']        
        print('TESTTT')
        print('ORIGINAL MODEL OUTPUT: ', model_output)
        
        if model_output is None or is_empty_list(model_output):
            return {'model_output':'[{"entity":""}]'}
        
        model_output = self._remove_json_special_chars(model_output)
                
        if are_entities_extracted_as_dict_keys_instead_of_values(model_output, example):
            print('ENTITIES EXTRACTED AS DICT KEYS INSTEAD OF VALUES')
            tmp = json.loads(model_output)
            tmp = [{"entity":k} for el in tmp for k in el.keys() ]
            tmp = str(tmp)
            return {'model_output':tmp}

        if is_numeric(model_output):
            print('IS NUMERIC')
            return {'model_output':'[{"entity":""}]'}
        
        if is_list_of_lists_and_dict(model_output):
            print('is_list_of_lists_and_dict')
            tmp = json.loads(model_output)
            for el in tmp:
                if isinstance(el, list):
                    tmp = str(el)
                    # print('is_list_of_lists_and_dict')
                    return {'model_output':tmp}
                
        if is_list_of_lists(model_output):
            print('is_list_of_lists')
            tmp = json.loads(model_output)
            tmp2 = str(tmp[0]).replace("'", "\"")
            if is_list_of_dicts_and_strings(tmp2):
                tmp = tmp[0]
                out = [item for item in tmp if isinstance(item, dict)]
                print('questo:', out)
                return {'model_output':str(out)} 
            tmp = str(tmp[0])
            return {'model_output':tmp}

        if is_list_of_strings(model_output):
            print('is_list_of_strings')
            tmp = json.loads(model_output)
            tmp = [{"entity":el} for el in tmp]
            tmp = str(tmp)
            # print('is_list_of_strings')
            return {'model_output': tmp}
        
        if is_string(model_output):
            # model_output = model_output.replace("{\'", "{\"").replace("\'}", "\"}").replace("\'ent", "\"ent").replace("ty\'", "ty\"").replace(" \'", " \"")
            print('PULITO: ', model_output)
            tmp = json.loads(model_output)
            if all(el in tmp for el in ['{', 'entity', '}']):
                return {'model_output':tmp}
            tmp = [{"entity":tmp}]
            tmp = str(tmp)
            #print('is_string')
            return {'model_output':tmp}

        
        if latest_version:
            model_output = self._extract_text_between_curl_brackets(model_output)
            #last attempt to clean 
            model_output = self._clean_text_between_curl_brackets(model_output)
            cleaning_done, cleaned_model_output = only_dicts_with_key_entity(model_output, wrong_keys_to_entity=wrong_keys_to_entity)
            if cleaning_done:
                model_output = cleaned_model_output
            
            if is_list_of_dicts(model_output):
                tmp = json.loads(model_output)
                #tmp = [item for item in tmp if 'entity' in item.keys()]
                return {'model_output':str(tmp)}
            
            else: 
                # print('NOT CLEANED: ', model_output, '\n\n')
                return {'model_output':'[{"entity":""}]'}
        

        if not latest_version:
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

            # print('THIS IS A BROKEN ROW: ', model_output)

            return {'model_output':model_output}
    
    def _extract_text_between_curl_brackets(self, model_output: str) -> str:
        """
        Extract the text between the curl brackets of the model output, as enitties are usually outputted in this format: {"entity": "value"}

        Args:
        model_output (str): the example from the dataset

        """
        text_between_curl_brackets = re.findall(r'\{(.+?)\}', model_output)
        cleaned_output = ['{'+ el +'}' for el in text_between_curl_brackets]
        cleaned_output = '[' + ', '.join(cleaned_output) + ']'
        return cleaned_output
    

    def _clean_text_between_curl_brackets(self, text_between_curl_brackets: str) -> str:
        """
        Clean the text between the curl brackets of the model output, as entities are usually outputted in this format: {"key": "value"}

        Args:
        model_output (str): the example from the dataset

        """
        text_between_curl_brackets = re.sub(r'",(.+?)}', r'"}', text_between_curl_brackets)
        return(text_between_curl_brackets)

        # list_of_dicts = json.loads(text_between_curl_brackets)
        # list_of_dicts_str = [str(item) for item in list_of_dicts]
        # out = []
        # print('list_of_dicts: ', list_of_dicts_str)
        # if ' '.join(list_of_dicts_str).count(':') > len(list_of_dicts):
        #     for item in list_of_dicts:
        #         if len(item) > 1:
        #             item = self._keep_only_one_keyVal_from_dict(item)
        #         out.append(item)
        #return str(out)

    # def _keep_only_one_keyVal_from_dict(dict: dict) -> dict:
    #     """
    #     Keep only one key-value pair from a dictionary. This is useful when the model outputs a dictionary with more than one key-value pair.

    #     Args:
    #     dict (dict): the dictionary to be cleaned

    #     return:
    #     dict: the cleaned dictionary
    #     """
    #     k_first = "entity"
    #     if k_first not in dict.keys():
    #         k_first = list(dict.keys())[0]
    #     return {k_first:dict[k_first]}

    
    def apply_cleaning(self, data, wrong_keys_to_entity) -> None:
        """
        Apply the cleaning to the model output and return the cleaned response in a new cloumn called 'model_output

        Args:
        model_output (str): the model output as it is returned by the model. The processing of the output is done in the function
        wrong_keys_to_entity (bool): if True, the function also extracts the dictionaries with keys different from 'entity', converting the keys into 'entity'. If not, all keys that are not 'entity' are dropped

        return:
        str: the model response, i.e. the model output without the instruction
        """
        data = data.map(lambda x: self._clean_ground_truth(x), remove_columns=['ground_truth'])
        data = data.map(lambda x: self._clean_model_output(x, wrong_keys_to_entity)) 
        return data