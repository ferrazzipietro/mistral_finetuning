import json
import re
import pandas as pd
from datasets import Dataset        
from fuzzywuzzy import fuzz


class Evaluator():

    def __init__(self,  data: Dataset, offset:bool, output_cleaner) -> None:
        self.offset = offset
        self.data = data
        self.cleaner = output_cleaner
        pass


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
        

    def _change_apexes(self, model_output: str) -> str:
        """
        Extract the text between the curl brackets of the model output, and change the apexes from double single \'.

        Args:
        model_output (str): the example from the dataset

        """
        text_between_curl_brackets = re.findall(r'\{(.+?)\}', model_output)
        cleaned_output = '['
        for el in text_between_curl_brackets:
            key_part, value_part = el.split(': ', 1)
            first_occurrence = value_part.find('"')
            last_occurrence = value_part.rfind('"')
            tmp = '{' + key_part + ': "' + value_part[first_occurrence+1:last_occurrence].replace('"', "'") + '"' + '}'
            cleaned_output += tmp + ', '
        cleaned_output = cleaned_output[:-2] + ']'
        return cleaned_output
    
    def _assess_model_output(self, model_response: str) -> (bool, str):
        """
        Check if the model output is in the right format. If not, return False.
        
        Args:
        model_output (str): the postprocessed model output after beeing passed to _postprocess_model_output()

        return:
        bool: True if the format is correct, False otherwise
        str: the model output in the adjusted format
        """
        good_format = True
        # if self.cleaner.verbose: print('prima sostituz:  ', model_response)
        model_response = model_response.replace("{\'", "{\"").replace("\'}", "\"}").replace("\'ent", "\"ent").replace("ty\'", "ty\"").replace(": \'", ": \"")
        model_response = re.sub(r'(?<=[a-zA-Z])"(?=[a-zA-Z])', "'", model_response)
        # if self.cleaner.verbose: print('dopo sostituz: ', model_response)
        try :
            out = json.loads(model_response)
            if isinstance(out, dict):
                model_response = '[' + model_response + ']'
        except Exception as error:
            if hasattr(error, 'msg'):
                if error.msg.startswith('Expecting property name enclosed in double quotes'):
                    model_response = model_response.replace("{\'", "{\"").replace("\'}", "\"}").replace("\'ent", "\"ent").replace("ty\'", "ty\"").replace(": \'", ": \"")
                    try:
                        out = json.loads(model_response)
                        if isinstance(out, dict):
                            model_response = '[' + model_response + ']'
                            good_format = True
                    except Exception as error2:
                        if isinstance(error2, json.decoder.JSONDecodeError):
                            if error2.msg == "Expecting ',' delimiter":
                                model_response = self._change_apexes(model_response)
                                good_format = True
            else:
                #print('MODEL RESPNSE: ', model_response)
                good_format = False
        if not good_format:
            model_response = re.findall(r'\[\{(.+?)\}\]', model_response)
            if len(model_response) != 0:
                model_response = '[{' + model_response[0] + '}]'
                good_format = True
                try :
                    json.loads(model_response)
                except Exception as error:
                    good_format = False
            else:
                good_format = False
        return good_format, model_response

    def _parse_json(self, model_response: str, drop_duplicates: bool = True) -> dict:
        """
        Parse the model output to extract the entities and their offsets if present.
        
        Args:
        model_response (str): the model response 
        drop_duplicates (bool): if True, drop the duplicates in the model response
        """
        # print('MODEL RESPONSE 1: ', model_response)
        model_response = model_response.replace("\n", " ")

        good_format, model_response = self._assess_model_output(model_response)

        # print('MODEL RESPONSE 2: ', model_response)        
        # print('MODEL RESPONSE 3: ', model_response)
        if model_response == []:
            model_response = '[{"entity":""}]'
        if self.offset and good_format:
            output = json.loads(model_response)
            if drop_duplicates:
                output = self._drop_duplicates(output)
            entities = [entity["entity"] for entity in output]
            offsets = [entity["offset"] for entity in output]
            return {"entities": entities, "offsets": offsets}
        if (not self.offset) and good_format:

            # print('MODEL RESPONSE 4: ', model_response)
            # print('ORA STO PARSANDO: ', model_response)
            output = json.loads(model_response)
            # print('OUTPUT: ', type(output))
            if drop_duplicates:
                output = self._drop_duplicates(output)
            entities = [entity["entity"].strip() for entity in output]
            # print('ENTITIES: ', entities)
            return {"entities": entities}
        if not good_format:
            return {"entities": []}
    
    def _count_common_words(self, string1: str, string2: str) -> int:
        """
        Count the number of common words between two entities without considering repetition.

        Args:
        string1 (str): an entity in the model response
        string2 (str): an entity in the ground truth

        return:
        int: the number of common words
        """
        model_words = set(string1.lower().split())
        ground_truth_words = set(string2.lower().split())
        common_words = model_words.intersection(ground_truth_words)
        return len(common_words)
        
    def _entity_similar_to_ground_truth_entity_LowerUppercase(self, entity_in_model_response: str, entity_in_ground_truth: str) -> (bool, str):
        """
        Check if two entities are similar, i.e. if the difference is just a fact of being upper or lower case.

        Args:
        entity_in_model_response (str): an entity in the model response
        entity_in_ground_truth (str): an entity in the ground truth
        threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely different, 100 is the same.

        return:
        bool: True if the entities are similar, False otherwise
        str: the entity in the ground truth if the entities are similar, the entity in the model response otherwise

        """
        FP_words = 0
        FN_words = 0
        TP_words = 0
        if entity_in_model_response.lower() == entity_in_ground_truth.lower():
            # print('SIMILI CASE: ', entity_in_model_response, ' e ', entity_in_ground_truth)
            TP_words = len(entity_in_ground_truth.split())
            return True, entity_in_ground_truth, FP_words, FN_words, TP_words
        return False, entity_in_model_response, FP_words, FN_words, TP_words
    
    def _entity_similar_to_ground_truth_entity_StopWords(self, entity_in_model_response: str, entity_in_ground_truth: str) -> (bool, str):
        """
        Check if two entities are similar, i.e. if the difference is just a stop words (e.g., "the" or "a"). Everything is performend in lower case.
        This is useful when the model output is not exactly the same as the ground truth.

        Args:
        entity_in_model_response (str): an entity in the model response
        entity_in_ground_truth (str): an entity in the ground truth
        threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely different, 100 is the same.

        return:
        bool: True if the entities are similar, False otherwise
        str: the entity in the ground truth if the entities are similar, the entity in the model response otherwise
        """
        def __preprocess_string__(string):
            # Remove common articles and noise words
            noise_words = ["a", "an", "the", "of"]
            words = string.split()
            filtered_words = [word for word in words if word.lower() not in noise_words]
            return ' '.join(filtered_words)
        
        FP_words = 0
        FN_words = 0
        TP_words = 0
        normalized_string = __preprocess_string__(entity_in_model_response)
        normalized_entity_ground_truth = __preprocess_string__(entity_in_ground_truth)
        if normalized_string == normalized_entity_ground_truth:
            n_words_ground_truth = len(entity_in_ground_truth.split())
            n_words_model_response = len(entity_in_model_response.split())
            FP_words = max(0, n_words_model_response - n_words_ground_truth)
            FN_words = max(0, n_words_ground_truth - n_words_model_response)
            TP_words = self._count_common_words(entity_in_model_response, entity_in_ground_truth)
            #print('SIMILI NORMALIZED: ', entity_in_model_response, ' e ', entity_in_ground_truth, ' -> FP_words:', FP_words,' FN_words:', FN_words,'TP_words:', TP_words)
            return True, entity_in_ground_truth, FP_words, FN_words, TP_words
        return False, entity_in_model_response, FP_words, FN_words, TP_words

    def _entity_similar_to_ground_truth_entity_Subset(self, entity_in_model_response: str, entity_in_ground_truth: str, ground_truth:list) -> (bool, str, int, int):
        """
        Check if two entities are similar in terms of being a subset of the one in list. E.g., entity='am happy' ground truth='I am happy'.
        This is useful when the model output is not exactly the same as the ground truth.

        Args:
        entity_in_model_response (str): an entity in the model response
        entity_in_ground_truth (str): an entity in the ground truth

        return:
        bool: True if the entities are similar, False otherwise
        str: the entity in the ground truth if the entities are similar, the entity in the model response otherwise
        FP_words: number of identified false positive words, i.e. number of words identified as entity that are not in the ground truth
        FN_words: number of identified false positive words, always 0 in this case
        TP_words: number of identified true positive words
        """
        FP_words = 0
        FN_words = 0
        TP_words = 0
        if entity_in_model_response.lower() != entity_in_ground_truth.lower():
            if entity_in_model_response.lower() in entity_in_ground_truth.lower():
                # words_not_in_gt_entity = entity_in_ground_truth.lower().replace(entity_in_model_response.lower(), '')
                # w_are_in_other_entity = any([words_not_in_gt_entity in gt_entity for gt_entity in ground_truth])
                # if w_are_in_other_entity:
                #     TP_words = self._count_common_words(entity_in_model_response, entity_in_ground_truth)
                # else:
                FN_words = entity_in_ground_truth.strip().count(" ") - entity_in_model_response.strip().count(" ")
                TP_words = self._count_common_words(entity_in_model_response, entity_in_ground_truth)
                # print('SIMILI Subset: ', entity_in_model_response, ' e ', entity_in_ground_truth, ' -> FP_words:', FP_words,' FN_words:', FN_words,'TP_words:', TP_words)
                return True, entity_in_ground_truth, FP_words, FN_words, TP_words
        return False, entity_in_model_response, FP_words, FN_words, TP_words

    def _entity_similar_to_ground_truth_entity_Superset(self, entity_in_model_response: str, entity_in_ground_truth: str) -> (bool, str, int, int):
        """
        Check if two entities are similar in terms of being a super of the one in list. E.g., entity='I am very happy' ground truth='I am happy'.
        This is useful when the model output is not exactly the same as the ground truth.

        Args:
        entity_in_model_response (str): an entity in the model response
        entity_in_ground_truth (str): an entity in the ground truth
        threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely different, 100 is the same.

        return:
        bool: True if the entities are similar, False otherwise
        str: the entity in the ground truth if the entities are similar, the entity in the model response otherwise
        FP_words: number of identified false positive words, i.e. number of words identified as entity that are not in the ground truth
        FN_words: number of identified false positive words, always 0 in this case
        """
        FP_words = 0
        FN_words = 0
        TP_words = 0
        if entity_in_model_response.lower() != entity_in_ground_truth.lower():
            if entity_in_ground_truth.lower() in entity_in_model_response.lower():
                FP_words = entity_in_model_response.strip().count(" ") - entity_in_ground_truth.strip().count(" ")
                TP_words = self._count_common_words(entity_in_model_response, entity_in_ground_truth)
                # print('SIMILI Superset: ', entity_in_model_response, ' e ', entity_in_ground_truth, ' -> FP_words:', FP_words,' FN_words:', FN_words,'TP_words:', TP_words)
                return True, entity_in_ground_truth, FP_words, FN_words, TP_words
        return False, entity_in_model_response, FP_words, FN_words, TP_words


    def _entity_similar_to_ground_truth_entity_Leveshtein(self, entity_in_model_response: str, entity_in_ground_truth: str, threshold: int) -> (bool, str):
        """
        Check if two entities are similar in terms of Leveshtein distance. This is useful when the model output is not exactly the same as the ground truth.

        Args:
        entity_in_model_response (str): an entity in the model response
        entity_in_ground_truth (str): an entity in the ground truth
        threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely different, 100 is the same.

        return:
        bool: True if the entities are similar, False otherwise
        str: the entity in the ground truth if the entities are similar, the entity in the model response otherwise
        """
        similarity = fuzz.ratio(entity_in_model_response.lower(), entity_in_ground_truth.lower())
        if similarity >= threshold:
            # print('SIMILI LEVESHTEIN: ', entity_in_model_response, ' e ', entity_in_ground_truth)
            return True, entity_in_ground_truth
        return False, entity_in_model_response
    

    def _entity_similar_to_ground_truth_entity(self, entity_in_model_response: str, entity_in_ground_truth: str, leveshtein_threshold: int, similarity_types:list=['case', 'stop_words', 'subset', 'superset', 'leveshtein'], ground_truth:list=[]) -> (bool, str):
        """
        Check if two entities are similar. This is useful when the model output is not exactly the same as the ground truth.

        Args:
        entity_in_model_response (str): an entity in the model response
        entity_in_ground_truth (str): an entity in the ground truth
        leveshtein_threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely different, 100 is the same.
        similarity_types (list): the list of similarity types to consider. The default value is ['case', 'stop_words', 'subset', 'superset', 'leveshtein']

        return:
        bool: True if the entities are similar, False otherwise
        str: the entity in the ground truth if the entities are similar, the entity in the model response otherwise
        """
        FP_words = 0
        FN_words = 0
        TP_words = 0

        if entity_in_model_response == entity_in_ground_truth:
            TP_words = len(entity_in_model_response.split())
            return True, entity_in_ground_truth, FP_words, FN_words, TP_words
        
        if 'case' in similarity_types:
            similar, entity_to_output, FP_words, FN_words, TP_words = self._entity_similar_to_ground_truth_entity_LowerUppercase(entity_in_model_response, entity_in_ground_truth)
            #print('SIMILI CASE: ', similar, entity_to_output, FP_words, FN_words, TP_words)
            if similar:
                return similar, entity_to_output, FP_words, FN_words, TP_words
        if 'stop_words' in similarity_types:
            similar, entity_to_output, FP_words, FN_words, TP_words = self._entity_similar_to_ground_truth_entity_StopWords(entity_in_model_response, entity_in_ground_truth)
            #print('SIMILI STOP WORDS: ', similar, entity_to_output, FP_words, FN_words, TP_words)
            if similar:
                return similar, entity_to_output, FP_words, FN_words, TP_words
        if 'subset' in similarity_types:
            similar, entity_to_output, FP_words, FN_words, TP_words = self._entity_similar_to_ground_truth_entity_Subset(entity_in_model_response, entity_in_ground_truth, entity_in_ground_truth)
            #print('SIMILI SUBSET: ', similar, entity_to_output, FP_words, FN_words, TP_words)
            if similar:
                return similar, entity_to_output, FP_words, FN_words, TP_words
        if 'superset' in similarity_types:
            similar, entity_to_output, FP_words, FN_words, TP_words = self._entity_similar_to_ground_truth_entity_Superset(entity_in_model_response, entity_in_ground_truth)  
            #print('SIMILI SUPERSET: ', similar, entity_to_output, FP_words, FN_words, TP_words)
            if similar:
                return similar, entity_to_output, FP_words, FN_words, TP_words
        if 'leveshtein' in similarity_types:
            similar, entity_to_output = self._entity_similar_to_ground_truth_entity_Leveshtein(entity_in_model_response, entity_in_ground_truth, leveshtein_threshold)
            #print('SIMILI LEVESTAIN: ', similar, entity_to_output)
            if similar:
                FP_words, FN_words, TP_words = 0, 0, 0 # non calcolo FP, FN, TP per leveshtein
                return similar, entity_to_output, FP_words, FN_words, TP_words

        return False, entity_in_model_response, FP_words, FN_words, TP_words
    

    # def _entity_similar_to_ground_truth_entity_deprecated(self, entity_in_model_response: str, entity_in_ground_truth: str, threshold: int) -> (bool, str):
    #     """
    #     Check if two entities are similar. This is useful when the model output is not exactly the same as the ground truth.

    #     Args:
    #     entity_in_model_response (str): an entity in the model response
    #     entity_in_ground_truth (str): an entity in the ground truth
    #     threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely different, 100 is the same.

    #     return:
    #     bool: True if the entities are similar, False otherwise
    #     str: the entity in the ground truth if the entities are similar, the entity in the model response otherwise
    #     """
    #     def __preprocess_string__(string):
    #         # Remove common articles and noise words
    #         noise_words = ["a", "an", "the", "of"]
    #         words = string.split()
    #         filtered_words = [word for word in words if word.lower() not in noise_words]
    #         return ' '.join(filtered_words)

    #     if entity_in_model_response == entity_in_ground_truth:
    #         return True, entity_in_ground_truth
        
    #     normalized_string = __preprocess_string__(entity_in_model_response)
    #     normalized_entity_ground_truth = __preprocess_string__(entity_in_ground_truth)
    #     if normalized_string == normalized_entity_ground_truth:
    #         print('SIMILI NORMALIZED 1: ', entity_in_model_response, ' e ', entity_in_ground_truth)
    #         return True, entity_in_ground_truth
        
    #     similarity = fuzz.ratio(entity_in_model_response.lower(), entity_in_ground_truth.lower())
    #     if similarity >= threshold:
    #         print('SIMILI LEVESTAIN 2: ', entity_in_model_response, ' e ', entity_in_ground_truth)
    #         return True, entity_in_ground_truth
    #     return False, entity_in_model_response
    
        
    def entity_in_ground_truth_list(self, entity_in_model_response: str, ground_truth: list, model_response_list: list, leveshtein_threshold: int, similarity_types:'list[str]') -> (str, int, int):
        """
        Check if an entity is in the ground truth

        Args:
        entity_in_model_response (str): an entity in the model response
        ground_truth (list): the ground truth
        model_response_list (list): the list off all entities already in the answer
        threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely different, 100 is the same.
        similarity_types: the list of similarity types to consider. Must contain elements in ['case', 'stop_words', 'subset', 'superset', 'leveshtein']

        return:
        bool: True if the entity is in the ground truth, False otherwise
        str: the entity in the ground truth if the entity is in the ground truth, the entity in the model response otherwise
        """
        strings = []
        FPs = []
        FNs = []
        TPs = []
        #print('\n ENTITY IN MODEL RESPONSE: ', entity_in_model_response)
        # if entity_in_model_response in ground_truth:
        #     print('TROVATO: ', entity_in_model_response)
        #     FPs.append(0)
        #     FNs.append(0)
        #     TPs.append(len(entity_in_model_response.split()))
        for entity_in_ground_truth in ground_truth:
            is_in, string, FP, FN, TP = self._entity_similar_to_ground_truth_entity(entity_in_model_response, entity_in_ground_truth, leveshtein_threshold, similarity_types, ground_truth)
            if is_in:
                strings.append(string)
                FPs.append(FP)
                FNs.append(FN)
                TPs.append(TP)
                continue
        #if entity_in_model_response in ground_truth and entity_in_model_response
        #print('STRINGS: ', strings)
        if len(strings) > 0:
            if entity_in_model_response in strings: # se ho estratto la stessa, ritorno se stessa
                TP = len(entity_in_model_response.split())
                #print('returning: ', entity_in_model_response, ' con TP: ', TP)
                return entity_in_model_response, 0, 0, TP
            else: #
                #print('sto analizzando: "', entity_in_model_response, '" e ho trovato: ', strings)

                #print('returning: ', entity_in_model_response, ' con TP: ', TP, ' FP: ', FP, ' FN: ', FN)
                return strings[-1], FPs[-1], FNs[-1], TPs[-1]
        else:
            FP = len(entity_in_model_response.split())
            FN = 0
            TP = 0
           
        #print('returning: ', entity_in_model_response, ' con TP: ', TP, ' FP: ', FP, ' FN: ', FN)
        return entity_in_model_response, FP, FN, TP
    


    def _extract_TP_FP_FN(self, model_response: str, ground_truth: str, similar_is_equal:bool=True, similar_is_equal_threshold: int=100, similarity_types:'list[str]'=['case', 'stop_words', 'subset', 'superset', 'leveshtein'], words_level:bool=True, already_parsed_inputs:bool=False) -> [int, int, int]:
        """
        Compute the F1 score, the precision and the recall between the model output and the ground truth

        Args:
        model_response (str): the model output as it is returned by the model
        ground_truth (str): the ground truth in json format.
        similar_is_equal (bool): if True, the function will consider similar entities as equal. The default value is False.
        similar_is_equal_threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely different, 100 is the same.
        words_level (bool): if True, the function will consider as base elements the words. If False, the function will consider as base elements the entity. 
        E.g., if True, the original sentence is "Yesterday morning I was so very happy and sad", the ground truth is ["yesterday morning", "so very happy"] the model output is ["morning", "happy and"], the function will consider FP=2 ("and"); TP=1 ("morning", "happy"); FN=2 ("Yesterday", "so"). 
        If False, the function will consider FP=1 ("happy and"); TP=0; FN=2 ("Yesterday morning", "so very happy").
        similarity_types: the list of similarity types to consider. Must contain elements in ['case', 'stop_words', 'subset', 'superset', 'leveshtein']

        """
        # print('MODEL RESPONSE: ', model_response)
        # print('GROUND TRUTH: ', ground_truth)
        if not already_parsed_inputs:
            # print('ORIGINAL model_response: ', model_response)
            model_response = self._parse_json(model_response)
            # print('GROUND TRUTH: ', ground_truth)
            ground_truth = self._parse_json(ground_truth.replace('\n', ''))
        model_response = model_response["entities"]
        ground_truth = ground_truth["entities"]
        
        
        # print('PARSED ORIGINAL model_response: ', model_response)
        # print('PARSED GROUND TRUTH: ', ground_truth)
        if not similar_is_equal:
            similarity_types = []

        if words_level:
            FP_sum = 0
            FN_sum = 0
            TP_sum = 0
            identified_entities_list = []
            for i, response_entity in enumerate(model_response):
                entity_identified, FP, FN, TP= self.entity_in_ground_truth_list(response_entity, ground_truth, model_response, similar_is_equal_threshold, similarity_types)
                FP_sum += FP
                FN_sum += FN
                TP_sum += TP
                identified_entities_list.append(entity_identified)
            #print('IDENTIFIED ENTITIES: ', identified_entities_list)
            #print('GROUND TRUTH: ', ground_truth)
            FN_entities = set(ground_truth).difference(set(identified_entities_list))
            FN_entities = [entity.split() for entity in FN_entities]
            FN_entities = [item for row in FN_entities for item in row]
            #print('FALSE NEGATIVES: ', FN_entities)
            FN_sum += len(FN_entities)
            #print('PARSED GROUND TRUTH: ', ground_truth, 'TP_sum:', TP_sum, 'FP_sum:', FP_sum, 'FN_sum:', FN_sum, '\n\n')
            return [TP_sum, FP_sum, FN_sum]
           
        elif not words_level:
            for i, response_entity in enumerate(model_response):
                model_response[i], _, _, _= self.entity_in_ground_truth_list(response_entity, ground_truth, model_response, similar_is_equal_threshold, similarity_types)
            #print('PARSED GROUND TRUTH: ', ground_truth)
            #print('NEW model_response to calculate TP, FP, FN: ', model_response, '\n\n')

            TP = len(set(model_response).intersection(set(ground_truth)))
            FP = len(set(model_response).difference(set(ground_truth)))
            FN = len(set(ground_truth).difference(set(model_response)))
            # F1 = 2 * TP / (2 * TP + FN + FP)
            return [TP, FP, FN]
    
    def generate_evaluation_table(self, similar_is_equal_threshold: int, words_level:bool, similarity_types:'list[str]', already_parsed_inputs:bool=False, add_TP_FP_TN_FN_to_data:bool=False) -> dict:
        """
        Generate the evaluation table for the model output and the ground truth.

        Args:
        similar_is_equal_threshold (int): the threshold to consider the entities similar by the Leveshtein distance. The default value is 80. 0 is completely
        different, 100 is the same. 
        words_level (bool): if True, the function will consider as base elements the words. If False, the function will consider as base elements the entity. 
        E.g., if True, the original sentence is "Yesterday morning I was so very happy and sad", the ground truth is ["yesterday morning", "so very happy"] the model output is ["morning", "happy and"], the function will consider FP=2 ("and"); TP=1 ("morning", "happy"); FN=2 ("Yesterday", "so"). 
        If False, the function will consider FP=1 ("happy and"); TP=0; FN=2 ("Yesterday morning", "so very happy").
        similarity_types: the list of similarity types to consider. Must contain elements in ['case', 'stop_words', 'subset', 'superset', 'leveshtein']

        return:
        dict: the evaluation table
        """
        self.evaluation_table = None
        metrics_list = []
        if not already_parsed_inputs:
            for i, res in enumerate(self.data['model_output']):
            #  if self.cleaner.verbose: print('res:', res)
                metrics_list.append(self._extract_TP_FP_FN(res, self.data['ground_truth'][i], True, similar_is_equal_threshold, similarity_types, words_level, already_parsed_inputs))
        else:
            for i, res in enumerate(self.data['model_output_parsed']):
                metrics_list.append(self._extract_TP_FP_FN(res, self.data['ground_truth_parsed'][i], True, similar_is_equal_threshold, similarity_types, words_level, already_parsed_inputs))
        metrics_dataframe = pd.DataFrame(metrics_list, columns=['TP', 'FP', 'FN'])
        summary = metrics_dataframe.sum()
        precision = summary['TP'] / (summary['TP'] + summary['FP'])
        recall = summary['TP'] / (summary['TP'] + summary['FN'])
        f1 = 2 * (precision * recall) / (precision + recall)
        self.evaluation_table = {'evaluation': metrics_dataframe, 'precision':precision, 'recall':recall, 'f1':f1}
        if add_TP_FP_TN_FN_to_data: self.add_TP_FP_TN_FN_to_data(already_parsed_inputs=already_parsed_inputs)

        return {'evaluation': metrics_dataframe, 'precision':precision, 'recall':recall, 'f1':f1}

    def add_TP_FP_TN_FN_to_data(self, already_parsed_inputs:bool=False):
        """
        Add the True Positives, False Positives, False Negatives, Precision, Recall and F1 score to the data. Precision, recall and f1 are approximated to avoid division by zero.
        """
        metrics_list = []
        if not already_parsed_inputs:
            for i, res in enumerate(self.data['model_output']):
                metrics_list.append(self._extract_TP_FP_FN(res, self.data['ground_truth'][i], True, 100, ['case', 'subset', 'superset'], already_parsed_inputs=already_parsed_inputs))
        else:
            for i, res in enumerate(self.data['model_output_parsed']):
                metrics_list.append(self._extract_TP_FP_FN(res, self.data['ground_truth_parsed'][i], True, 100, ['case', 'subset', 'superset'], already_parsed_inputs=already_parsed_inputs))
        if 'TP' in self.data.column_names:
            self.data = self.data.remove_columns(['TP', 'FP', 'FN', 'precision', 'recall', 'f1'])
        self.data = self.data.add_column('TP', [el[0] for el in metrics_list])
        self.data = self.data.add_column('FP', [el[1] for el in metrics_list])
        self.data = self.data.add_column('FN', [el[2] for el in metrics_list])
        self.data = self.data.add_column('precision', [el[0] / (el[0] + el[1] + 1e-16) for el in metrics_list])
        self.data = self.data.add_column('recall', [el[0] / (el[0] + el[2] + 1e-16) for el in metrics_list])
        def _f1(example):
            example['f1'] = 2 * (example['precision'] * example['recall']) / (example['precision'] + example['recall'] + 1e-16)
            return example
        self.data = self.data.map(lambda x: _f1(x))