from utils.data_preprocessor import DataPreprocessor
from config import preprocessing_params
from datasets import Dataset
from tqdm import tqdm
import json
import re

class OutputCleaner():
    def __init__(self, test_data: Dataset, preprocessor:DataPreprocessor, n_shots_inference:int, language:str, tokenizer) -> None:
        self.test_data = test_data
        self.preprocessor = preprocessor
        self.language = language
        self.tokenizer = tokenizer
        self.few_shots_dict = {'en':{'questions':['We present a case of a 32-year-old woman with a history of gradual enlargement of the anterior neck.',
                                                   'Patient information: a 9-month-old boy presented to the emergency room with a 3-day history of refusal to bear weight on the right lower extremity and febrile peaks of up to 38.5°C for 24 hours.'],
                                        'responses':['[{"entity": "present"}, {"entity": "history"}, {"entity": "enlargement"}]',
                                                     '[{"entity": "presented"}, {"entity": "refusal"}, {"entity": "bear"}, {"entity": "peaks"}]'],
                                        'responses_offset': ['[{"entity": "present", "offset": [3, 10]}, {"entity": "history", "offset": [48, 55]}, {"entity": "enlargement", "offset": [67, 78]}]',
                                                             '[{"entity": "presented", "offset": [39, 48]}, {"entity": "refusal", "offset": [95, 102]}, {"entity": "bear", "offset": [106, 110]}, {"entity": "peaks", "offset": [159, 164]}]']
                                    },
                                'it':{'questions':['In considerazione dell’inefficacia della terapia somministrata, in assenza di ulteriori opzioni terapeutiche standard potenzialmente efficaci e dopo colloquio con i genitori si decide di avviare la paziente a trapianto aploidentico, possibilmente NK allo reattivo, da genitore.',
                                                    'L’esame istologico dimostrava mucosa gastrica atrofica con flogosi cronica, marcato edema ed incremento del connettivo del corion, focale metaplasia intestinale, il tutto sovrastante un tessuto fibromuscolare.'],
                                       'responses':['[{"entity": "inefficacia"}, {"entity": "opzioni"}, {"entity": "colloquio"}, {"entity": "avviare"}, {"entity": "trapianto"}, {"entity": "genitori"}, {"entity": "paziente"}, {"entity": "genitore"}]',
                                                    ],
                                       'responses_offset':['[{"entity": "inefficacia", "offset": [23, 34]}, {"entity": "opzioni", "offset": [88,95]}, {"entity": "colloquio", "offset": [149,158]}, {"entity": "avviare", "offset": [187,194]}, {"entity": "trapianto", "offset": [209,218]}, {"entity": "genitori", "offset": [163,173]}, {"entity": "paziente", "offset": [195,106]}, {"entity": "genitore", "offset": [268,276]}]',
                                                           '[{"entity": "mucosa gastrica atrofica", "offset": [30,54]}, {"entity": "flogosi\r\cronica", "offset": [59,75]}]']}
                                }
        if len(self.few_shots_dict[self.language]['questions']) < n_shots_inference:
            raise ValueError(f'The number of shots for the inference prompt is greater than the number of examples available.')
        if len(self.few_shots_dict[self.language]['responses']) < n_shots_inference:
            raise ValueError(f'The number of shots for the inference prompt is greater than the number of responses available.')
        self.n_shots_inference = n_shots_inference

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
  