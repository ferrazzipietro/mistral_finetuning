import json
import re
import pandas as pd
from datasets import Dataset        
from fuzzywuzzy import fuzz


class Evaluator():

    def __init__(self,  data: Dataset, offset:bool) -> None:
        self.offset = offset
        self.data = data
        pass
    
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
        try :
            out = json.loads(model_response)
            if isinstance(out, dict):
                model_response = '[' + model_response + ']'
        except Exception as error:
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
        good_format, model_response = self._assess_model_output(model_response)
        if model_response == []:
            model_response = "[{'entity':''}]"
        if self.offset and good_format:
            output = json.loads(model_response)
            if drop_duplicates:
                output = self._drop_duplicates(output)
            entities = [entity["entity"] for entity in output]
            offsets = [entity["offset"] for entity in output]
            return {"entities": entities, "offsets": offsets}
        elif (not self.offset) and good_format:
            output = json.loads(model_response)
            if drop_duplicates:
                output = self._drop_duplicates(output)
            entities = [entity["entity"] for entity in output]
            return {"entities": entities}
        if not good_format:
            return {"entities": []}
        
    def _drop_duplicates(self, model_response: list) -> str:
        """
        Drop the duplicates from a list. This is useful when the model output contains the same entity multiple times.

        Args:
        model_response (str): the model response with no duplicates
        """
        return list({v['entity']:v for v in model_response}.values())


    def _entity_similar_to_ground_truth_entity(self, entity_in_model_resonse: str, entity_in_ground_truth: str, threshold: int) -> (bool, str):
        """
        Check if two entities are similar. This is useful when the model output is not exactly the same as the ground truth.

        Args:
        entity_in_model_resonse (str): an entity in the model response
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

        if entity_in_model_resonse == entity_in_ground_truth:
            return True, entity_in_ground_truth
        
        normalized_string = __preprocess_string__(entity_in_model_resonse)
        normalized_entity_ground_truth = __preprocess_string__(entity_in_ground_truth)
        if normalized_string == normalized_entity_ground_truth:
            # print('entity_ground_truth:', entity_in_ground_truth)
            return True, entity_in_ground_truth
        
        similarity = fuzz.ratio(entity_in_model_resonse.lower(), entity_in_ground_truth.lower())
        if similarity >= threshold:
            #print('VERO 2')
            return True, entity_in_ground_truth
        return False, entity_in_model_resonse
        
    def entity_in_ground_truth_list(self, entity_in_model_resonse: str, ground_truth: list, model_response_list: list, threshold: int) -> (bool, str):
        """
        Check if an entity is in the ground truth. This is useful when the model output is not exactly the same as the ground truth.

        Args:
        entity_in_model_resonse (str): an entity in the model response
        ground_truth (list): the ground truth
        model_response_list (list): the list off all entities already in the answer
        threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely different, 100 is the same.

        return:
        bool: True if the entity is in the ground truth, False otherwise
        str: the entity in the ground truth if the entity is in the ground truth, the entity in the model response otherwise
        """
        model_response_list
        strings = []
        for entity_in_ground_truth in ground_truth:
            is_in, string = self._entity_similar_to_ground_truth_entity(entity_in_model_resonse, entity_in_ground_truth, threshold)
            if is_in:
                strings.append(string)
        #if entity_in_model_resonse in ground_truth and entity_in_model_resonse
        if len(strings) > 0:
            if entity_in_model_resonse in strings: # se ho estratto la stessa, ritorno se stessa
                return entity_in_model_resonse
            else: #
                # print('sto analizzando: "', entity_in_model_resonse, '" e ho trovato: ', strings)
                return strings[-1]
        return entity_in_model_resonse
    


    def _extract_TP_FP_FN(self, model_response: str, ground_truth: str, similar_is_equal:bool, similar_is_equal_threshold: int) -> [int, int, int]:
        """
        Compute the F1 score, the precision and the recall between the model output and the ground truth

        Args:
        model_output (str): the model output as it is returned by the model. The processing of the output is done in the function
        ground_truth (str): the ground truth in json format.
        similar_is_equal (bool): if True, the function will consider similar entities as equal. The default value is False.
        similar_is_equal_threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely different, 100 is the same.

        """
        model_response = self._parse_json(model_response)
        ground_truth = self._parse_json(ground_truth)
        model_response = model_response["entities"]
        ground_truth = ground_truth["entities"]
        # print('ORIGINAL model_response: ', model_response)
        if similar_is_equal:
            for i, response_entity in enumerate(model_response):
                model_response[i] = self.entity_in_ground_truth_list(response_entity, ground_truth, model_response, similar_is_equal_threshold)
        # print('GROUND TRUTH: ', ground_truth)
        # print('NEW model_response: ', model_response, '\n\n')

        TP = len(set(model_response).intersection(set(ground_truth)))
        FP = len(set(model_response).difference(set(ground_truth)))
        FN = len(set(ground_truth).difference(set(model_response)))
        # F1 = 2 * TP / (2 * TP + FN + FP)
        return [TP, FP, FN]
    
    def generate_evaluation_table(self, similar_is_equal:bool, similar_is_equal_threshold: int) -> dict:
        """
        Generate the evaluation table for the model output and the ground truth.

        Args:
        similar_is_equal (bool): if True, the function will consider similar entities as equal. The default value is False.
        similar_is_equal_threshold (int): the threshold to consider the entities similar. The default value is 80. 0 is completely
        different, 100 is the same.

        return:
        dict: the evaluation table
        """
        metrics_list = []
        for i, res in enumerate(self.data['model_responses']):
            metrics_list.append(self._extract_TP_FP_FN(res, self.data['ground_truth'][i], similar_is_equal, similar_is_equal_threshold))

        metrics_dataframe = pd.DataFrame(metrics_list, columns=['TP', 'FP', 'FN'])
        summary = metrics_dataframe.sum()
        precision = summary['TP'] / (summary['TP'] + summary['FP'])
        recall = summary['TP'] / (summary['TP'] + summary['FN'])
        f1 = 2 * (precision * recall) / (precision + recall)
        self.evaluation_table = {'evaluation': metrics_dataframe, 'precision':precision, 'recall':recall, 'f1':f1}
        return {'evaluation': metrics_dataframe, 'precision':precision, 'recall':recall, 'f1':f1}