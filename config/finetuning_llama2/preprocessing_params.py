from transformers import AutoTokenizer
from .config import BASE_MODEL_CHECKPOINT

task='finetuning'
offset=False
instruction_on_response_format='Return the result in a json format: [{"entity":"entity_name"}].'
n_shots = 0
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT)
list_of_examples=[]
list_of_responses=[]

# The following are just examples
first_shot_example = 'We present a case of a 32-year-old woman with a history of gradual enlargement of the anterior neck.'
second_shot_example = 'Patient information: a 9-month-old boy presented to the emergency room with a 3-day history of refusal to bear weight on the right lower extremity and febrile peaks of up to 38.5°C for 24 hours.'
first_response = '[{"entity": "present", "offset": [3, 10]}, {"entity": "history", "offset": [48, 55]}, {"entity": "enlargement", "offset": [67, 78]}]'
second_response = '[{"entity": "presented", "offset": [39, 48]}, {"entity": "refusal", "offset": [95, 102]}, {"entity": "bear", "offset": [106, 110]}, {"entity": "peaks", "offset": [159, 164]}]'
input = "A 46-year-old man with hypertension and dyslipidemia diagnosed 4-months before, as well as new-onset diabetes mellitus unveiled 1-month earlier, was referred to emergency department for hypokalemia"
output = '[{"entity": "hypertension", "offset": [13, 25]}, {"entity": "dyslipidemia", "offset": [30, 42]}, {"entity": "diabetes mellitus", "offset": [74, 91]}, {"entity": "hypokalemia", "offset": [143, 154]}]'