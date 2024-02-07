import json
import re
from utils.data_preprocessor import DataPreprocessor
import pandas as pd
from datasets import Dataset

class Evaluator():

    def __init__(self, preprocessor:DataPreprocessor, data: Dataset) -> None:
        self.offset = preprocessor.offset
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
            json.loads(model_response)
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


    def _extract_TP_FP_FN(self, model_response: str, ground_truth: str) -> [int, int, int]:
        """
        Compute the F1 score, the precision and the recall between the model output and the ground truth

        Args:
        model_output (str): the model output as it is returned by the model. The processing of the output is done in the function
        ground_truth (str): the ground truth in json format.

        """
        model_response = self._parse_json(model_response)
        ground_truth = self._parse_json(ground_truth)
        model_response = model_response["entities"]
        ground_truth = ground_truth["entities"]
        TP = len(set(model_response).intersection(set(ground_truth)))
        FP = len(set(model_response).difference(set(ground_truth)))
        FN = len(set(ground_truth).difference(set(model_response)))
        # F1 = 2 * TP / (2 * TP + FN + FP)
        return [TP, FP, FN]
    
    def generate_evaluation_table(self):
        metrics_list = []
        for i, res in enumerate(self.data['model_responses']):
            metrics_list.append(self._extract_TP_FP_FN(res, self.data['ground_truth'][i]))

        metrics_dataframe = pd.DataFrame(metrics_list, columns=['TP', 'FP', 'FN'])
        summary = metrics_dataframe.sum()
        precision = summary['TP'] / (summary['TP'] + summary['FP'])
        recall = summary['TP'] / (summary['TP'] + summary['FN'])
        f1 = 2 * (precision * recall) / (precision + recall)
        return {'evaluation': metrics_dataframe, 'precision':precision, 'recall':recall, 'f1':f1}