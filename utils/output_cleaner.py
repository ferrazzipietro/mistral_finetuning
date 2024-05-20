import json
import re
from typing import Tuple
from typing import List

class OutputCleaner():
    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
  
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
            # print('DICT: ', dict)
            out.append({k.replace(' ', ''): v for k, v in dict.items()})
        return out
    
    def _drop_duplicates(self, model_response: list) -> str:
        """
        Drop the duplicates from a list. This is useful when the model output contains the same entity multiple times.

        Args:
        model_response (str): the model response with no duplicates
        """
        # print('DROPPING DUPLICATES: ', model_response)
        try :
            return list({v['entity']:v for v in model_response}.values())
        except Exception as error:
            model_response = self._remove_space_from_dict_keys(model_response)
            # print('ERROR: ', model_response)
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
        """
        Remove the special characters from a string. This is useful when the model output contains special characters that are not allowed in the json format.
        """
        # print('sto pulendo: ', string)
        chars = ['\xa0', '\x80', '\x93', '\U00100000', '\r\n', '\U00100000I', '\\u001d', '\\"']
        for char in chars:
            string = string.replace(char, ' ')
        char_no_space = ['\xad']
        for char in char_no_space:
            string = string.replace(char, '')
        string = string.replace('\\u0010', '^')
        return string
    
    def _special_cases_handler(self, model_response: str) -> str:
        """
        Handle the special cases in the model output. This is useful when the model output contains special characters that are not allowed in the json format.
        Ideally, this function should not be used. It is very specific for encountered issues I could not find a solution to.
        """
        #print('IN SPECIAL CASES HANDLER: ', model_response)
        model_response = model_response.replace(""" {"entity":"un\'insufficienza midollare\\" \\"- congenita"},""", "").\
            replace("""l\'aspetto\\"anteriorpuntale""", """l'aspetto anteriorpuntale""")
        model_response = model_response.replace("""rigonfiamento aneurismatico dell'apice del ventricolo sinistro\\\"""", """""").\
            replace('<|eot_id|>', '')
        return model_response
    
    def _clean_ground_truth(self, example: dict) -> dict:
        ground_truth = example['ground_truth']
        # print('inner ground truth: ', ground_truth)
        ground_truth = self._remove_json_special_chars(ground_truth)
        ground_truth = ground_truth.replace('</s>', '').replace('<|im_e', '').replace('<|end_of_text|>', '').replace('<|endoftext|>', '')
        if ground_truth.strip() == ']':
            ground_truth = '[]'
        # print('mid ground truth: ', ground_truth)
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

        def is_list_of_empty_dict(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                #print('TMP: ', tmp)
                if isinstance(tmp, list) and all(isinstance(item, dict) for item in tmp):
                    if all(str(item) == "{}" for item in tmp):
                        return True
            return False

        def is_list_with_one_empty_dict(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list):
                    for item in tmp:
                        if item == {}:
                            return True
            return False
        
        def is_list_of_dicts_with_empty_lists(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list) and all(isinstance(item, dict) for item in tmp):
                    for item in tmp:
                        for v in item.values():
                            if v == []:
                                return True
            return False
        
        def is_list_of_dicts_with_one_key_multiple_values(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list) and all(isinstance(item, dict) for item in tmp):
                    for item in tmp:
                        if len(item) == 1 and len(item.values()) > 1:
                            return True
            return False

        def is_list_of_dicts_with_multiple_keys_included_entity(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list) and all(isinstance(item, dict) for item in tmp):
                    for item in tmp:
                        if len(item) > 1 and 'entity' in item.keys():
                            return True
            return False

        def is_list_of_dict_numeric_values(string:str)  -> bool:
            #print('STRING: ', string)
            if self._assess_model_output(string):
                tmp = json.loads(string)
                #print('TMP: ', tmp)
                if isinstance(tmp, list) and all(isinstance(item, dict) for item in tmp):
                    for item in tmp:
                        if len(item.values()) > 0:
                            val = list(item.values())[0] 
                            if isinstance(val, int) or isinstance(val, float):
                                return True
            return False
        
        def is_list_of_dicts_none_values(string:str) -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list) and all(isinstance(item, dict) for item in tmp):
                    for item in tmp:
                        if len(item.values()) > 0:
                            val = list(item.values())[0] 
                            if val is None:
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
        
        def is_list_of_dicts_and_lists_of_strings(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                # print('TMP: ', tmp)
                if isinstance(tmp, list):
                    if all(isinstance(item, dict) for item in tmp):
                        return False
                    for item in tmp:
                        # print('ITEM: ', item)
                        if isinstance(item, dict):
                            
                            if len(item.values()) == 0:
                               return False
                            if item.get('entity') is None:
                                return False
                        elif isinstance(item, list):
                            if len(item) != 1:
                                return False
                            if not isinstance(item[0], str):
                                return False
                        else:
                            return False
                    return True
            return False
        
        def is_list_of_dicts_with_value_list(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, list) and all(isinstance(item, dict) for item in tmp):
                    for item in tmp:
                        for v in item.values():
                            if isinstance(v, list):
                                return True
            return False
        
        def is_string(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                if isinstance(tmp, str):
                    return True
            return False
        
        def is_list_of_strings_representing_dicts(string:str)  -> bool:
            if self._assess_model_output(string):
                tmp = json.loads(string)
                # print('TMP: ', tmp)
                if isinstance(tmp, list) and all(isinstance(item, str) for item in tmp):
                    tmp_list = []
                    for item in tmp:
                        # print('ITEM: ', item)
                        if self._assess_model_output(item):
                          tmp_list.append(json.loads(item))
                    if all(isinstance(item, dict) for item in tmp_list):
                        return True
            return False
        
        def is_list_of_dicts_of_lists(string:str)  -> bool:
            # print('STRING: ', string)
            if self._assess_model_output(string):
                tmp = json.loads(string)
                # print('TMP: ', tmp)
                if isinstance(tmp, list) and all(isinstance(item, dict) for item in tmp):
                    for item in tmp:
                        # print('item: ',item)
                        tmp2 = list(item.values())[0]
                        if len(tmp2) > 0:
                            if isinstance(list(item.values())[0], list):
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
            # out = out.replace("{\'", "{\"").replace("\'}", "\"}").replace("\'ent", "\"ent").replace("ty\'", "ty\"").replace(" \'", " \"")
            operations_performed = False
            if len(clean) != len(els_between_curly):
                operations_performed = True
            if is_empty_list(out):
                return operations_performed, '[{"entity":""}]'
            return operations_performed, str(out)
        
        if self.verbose: print('EXAMPLE:  ', example['model_responses'])
        model_output = example['model_responses']
        if self.verbose: print('ORIGINAL MODEL OUTPUT:', model_output)
        # print('ORIGINAL MODEL OUTPUT:', model_output)
        if self.verbose: print('GROUND TRUTH: ', example['ground_truth'])
        # model_output = self._exceptions_handler(model_output)
    
        if model_output is None or is_empty_list(model_output):
            return {'model_output':'[{"entity":""}]'}
        
        model_output = self._special_cases_handler(model_output)
        model_output = self._remove_json_special_chars(model_output)
        if self.verbose:print('PULITO: ', model_output)
                
        if are_entities_extracted_as_dict_keys_instead_of_values(model_output, example):
            if self.verbose: print('ENTITIES EXTRACTED AS DICT KEYS INSTEAD OF VALUES')
            tmp = json.loads(model_output)
            tmp = [{"entity":k} for el in tmp for k in el.keys() ]
            tmp = str(tmp)
            return {'model_output':tmp}
        
        if is_list_of_dicts_and_lists_of_strings(model_output):
            if self.verbose: print('is_list_of_dicts_and_lists_of_strings')
            tmp = json.loads(model_output)
            out = []
            for item in tmp:
                if self.verbose: print('ITEM: ', item)
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, list):
                    out.append({"entity":item[0]})
            return {'model_output':str(out)}

        if is_numeric(model_output):
            # print('IS NUMERIC')
            return {'model_output':'[{"entity":""}]'}

        # print('QUI HO QUESTO: ', model_output)
        if is_list_of_strings_representing_dicts(model_output):
            if self.verbose: print('is_list_of_strings_representing_dicts 1')                
            tmp = json.loads(model_output)
            tmp_list = []
            for item in tmp:
                if self._assess_model_output(item):
                  tmp_list.append(json.loads(item))
            if self.verbose: print('TEMPOOOO 2 ',tmp)
            return {'model_output':str(tmp_list)}
        
        if is_list_of_dicts_with_one_key_multiple_values(model_output):
            if self.verbose: print('is_list_of_dicts_with_one_key_multiple_values')
            tmp = json.loads(model_output)
            tmp = [{"entity":v[0]} for el in tmp for v in el.values()]
            return {'model_output':str(tmp)}
       
        if is_list_of_dicts_with_multiple_keys_included_entity(model_output):
            if self.verbose: print('is_list_of_dicts_with_multiple_keys_included_entity')
            tmp = json.loads(model_output)
            out = []
            for item in tmp:
                if item.get('entity') is not None:
                    out.append({"entity":item.get('entity')})
            return {'model_output':str(out)}
        
        
        if is_list_of_lists_and_dict(model_output):
            if self.verbose: print('is_list_of_lists_and_dict')
            tmp = json.loads(model_output)
            for el in tmp:
                if isinstance(el, list):
                    tmp = str(el)
                    # print('is_list_of_lists_and_dict')
                    return {'model_output':tmp}
                
        if is_list_of_lists(model_output):
            if self.verbose: print('is_list_of_lists')
            tmp = json.loads(model_output)
            tmp2 = str(tmp[0]).replace("'", "\"")
            if is_list_of_dicts_and_strings(tmp2):
                tmp = tmp[0]
                out = [item for item in tmp if isinstance(item, dict)]
                return {'model_output':str(out)} 
            tmp = str(tmp[0])
            return {'model_output':tmp}
        

        if is_list_of_strings(model_output):
            if self.verbose: print('is_list_of_strings')
            tmp = json.loads(model_output)
            tmp = [{"entity":el} for el in tmp]
            tmp = str(tmp)
            # print('is_list_of_strings')
            if self.verbose: print('TEMPOOOO ',tmp)
            return {'model_output': tmp}
        
        if is_string(model_output):
            # model_output = model_output.replace("{\'", "{\"").replace("\'}", "\"}").replace("\'ent", "\"ent").replace("ty\'", "ty\"").replace(" \'", " \"")
            if self.verbose: print('PULO: ', model_output)
            tmp = json.loads(model_output)
            if all(el in tmp for el in ['{', 'entity', '}']):
                return {'model_output':tmp}
            tmp = [{"entity":tmp}]
            tmp = str(tmp)
            #print('is_string')
            return {'model_output':tmp}

        
        if latest_version:
            model_output = self._extract_text_between_curl_brackets(model_output)
            model_output = self._clean_text_between_curl_brackets(model_output)

            # print('QUI HO il SECONDO QUESTO: ', model_output)

            if is_list_of_strings_representing_dicts(model_output):
                if self.verbose: print('is_list_of_strings_representing_dicts 2')                
                tmp = json.loads(model_output)
                tmp_list = []
                for item in tmp:
                    if self._assess_model_output(item):
                        tmp_list.append(json.loads(item))
                return {'model_output':str(tmp_list)}
            
            if is_list_of_dicts_with_one_key_multiple_values(model_output):
                if self.verbose: print('is_list_of_dicts_with_one_key_multiple_values')
                tmp = json.loads(model_output)
                tmp = [{"entity":v[0]} for el in tmp for v in el.values()]
                return {'model_output':str(tmp)}
            
            if is_list_of_dicts_and_lists_of_strings(model_output):
                if self.verbose: print('is_list_of_dicts_and_lists_of_strings')
                tmp = json.loads(model_output)
                out = []
                for item in tmp:
                    # print('ITEM: ', item)
                    if isinstance(item, dict):
                        out.append(item)
                    elif isinstance(item, list):
                        out.append({"entity":item[0]})
                return {'model_output':str(out)}
            
            if self.verbose: print('QUI HO il TEERZO QUESTO: ', model_output)

            if is_list_of_dicts_with_empty_lists(model_output):
                if self.verbose: print('is_list_of_dicts_with_empty_lists')
                tmp = json.loads(model_output)
                tmp = [{"entity":v} for el in tmp for v in el.values() if v != []]
                # print('TMP: ', tmp)
                if is_list_of_dicts_with_value_list(str(tmp).replace("'", "\"")):
                    if self.verbose: print('is_list_of_dicts_with_value_list')
                    tmp = [{"entity":v} for el in tmp for v in el.values() if not isinstance(v, list)]
                    tmp2 = [{"entity":v[0]} for el in tmp for v in el.values() if isinstance(v, list)]
                    # print('returning this: ', {'model_output ':str(tmp2)}  )
                    return {'model_output':str(tmp2)}
                # print('returning this: ', {'model_output ':str(tmp)}  )

                return {'model_output':str(tmp)}
            
            if self.verbose: print('QUI HO il QUARTO QUESTO:', model_output)

            if is_list_of_dicts_with_value_list(model_output):
                if self.verbose: print('is_list_of_dicts_with_value_list')
                tmp = json.loads(model_output)
                tmp = [{"entity":v} for el in tmp for v in el.values() if not isinstance(v, list)]
                tmp2 = [{"entity":v[0]} for el in tmp for v in el.values() if isinstance(v, list)]
                return {'model_output':str(tmp)}

            if is_list_of_dict_numeric_values(model_output):
                if self.verbose: print('is_list_of_dict_int_values')
                tmp = json.loads(model_output)
                tmp = [str({"entity":str(v)}) for el in tmp for v in el.values()]
                model_output = str(tmp)
            
            if is_list_of_dicts_none_values(model_output):
                if self.verbose: print('is_list_of_dicts_none_values')
                tmp = json.loads(model_output)
                tmp = [str({"entity":v}) for el in tmp for v in el.values() if v is not None]
                model_output = str(tmp)
                    
            if is_list_of_empty_dict(model_output):
                if self.verbose: print('is_list_of_empty_dict')
                return {'model_output':'[{"entity":""}]'}
            
            if is_list_with_one_empty_dict(model_output):
                if self.verbose: print('is_list_with_one_empty_dict')
                tmp = json.loads(model_output)
                tmp = [el for el in tmp if el != {}]
                model_output = tmp
                return {'model_output':str(model_output)}
            
            if is_list_of_dicts_of_lists(model_output):
                if self.verbose: print('is_list_of_dicts_of_lists')
                tmp = json.loads(model_output)
                tmp = [{"entity":v} for el in tmp for v in el.values() if not isinstance(v, list)]
                # tmp.extend([{"entity":el.values()[0]} for el in tmp if isinstance(el.values(), list)])
                # print('returning this: ', {'model_output ':str(tmp)}  )
                return {'model_output':str(tmp)}  
                
            if self.verbose: print('CLEANED: ', model_output)
            cleaning_done, cleaned_model_output = only_dicts_with_key_entity(model_output, wrong_keys_to_entity=wrong_keys_to_entity)
            if cleaning_done:
                model_output = cleaned_model_output
            
            if is_list_of_dicts(model_output):
                if self.verbose: print('PRE  CLEANED: ', model_output)
                if is_list_of_dicts_with_multiple_keys_included_entity(model_output):
                    if self.verbose: print('is_list_of_dicts_with_multiple_keys_included_entity')
                    tmp = json.loads(model_output)
                    out = []
                    for item in tmp:
                        if len(item) > 1 and 'entity' in item.keys():
                            out.append({"entity":item.get('entity')})
                    return {'model_output':str(out)}
                tmp = json.loads(model_output)
                return {'model_output':str(tmp)}
            
            else: 
                # print('NOT CLEANED: ', model_output, '\n\n')
                return {'model_output':'[{"entity":""}]'}
        
            
    def _exceptions_handler(self, model_output: str, error) -> str:
        # if hasattr(error, 'msg'):
        #     if error.msg.startswith('Expecting property name enclosed in double quotes'):
        #         model_output = model_output.replace("{\'", "{\"").replace("\'}", "\"}").replace("\'ent", "\"ent").replace("ty\'", "ty\"").replace(": \'", ": \"")
        
        try:
            json.loads(model_output)
        except Exception as error:
            if isinstance(error, json.decoder.JSONDecodeError):
                #if error.msg == "Expecting ',' delimiter":
                key_part, value_part = model_output.split(': ', 1)
                first_occurrence = value_part.find('"')
                last_occurrence = value_part.rfind('"')
                model_output = key_part + ': "' + value_part[first_occurrence+1:last_occurrence].replace("'", r'\'') + '"' + '}'
        return model_output
    # .replace("\'", " ")
    
    def _substitute_apexes(self, model_output: str) -> str:
        model_output = model_output.replace("{\'", "{\"").replace("\'}", "\"}").replace("\'ent", "\"ent").replace("ty\'", "ty\"").replace(": \'", ": \"")
        return model_output
    
    
    def _extract_text_between_curl_brackets(self, model_output: str) -> str:
        """
        Extract the text between the curl brackets of the model output, as enities are usually outputted in this format: {"entity": "value"}

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
        text_between_curl_brackets = re.sub(r'{},', r'', text_between_curl_brackets)
        text_between_curl_brackets = re.sub(r',{}', r'', text_between_curl_brackets)
        # print('CLEANED: ', text_between_curl_brackets)
        # text_between_curl_brackets = re.sub(r'\{"entity":\[\]\},', r'', text_between_curl_brackets)
        # text_between_curl_brackets = re.sub(r',{\'entity\':[]}', r'', text_between_curl_brackets)
        return text_between_curl_brackets
    
    def apply_cleaning(self, data, wrong_keys_to_entity) -> None:
        """
        Apply the cleaning to the model output and return the cleaned response in a new cloumn called 'model_output

        Args:
        data (list): the dataset containing the model output
        wrong_keys_to_entity (bool): if True, the function also extracts the dictionaries with keys different from 'entity', converting the keys into 'entity'. If not, all keys that are not 'entity' are dropped
        """
        data = data.filter(lambda example: example["entities"] is not None)
        data = data.map(lambda x: self._clean_ground_truth(x), remove_columns=['ground_truth'])
        data = data.map(lambda x: self._clean_model_output(x, wrong_keys_to_entity)) 
        return data