from transformers import AutoTokenizer
from .config import BASE_MODEL_CHECKPOINT

task='finetuning'
offset=False
instruction_on_response_format='Return the result in a json format: [{"entity":"entity_name"}].'
n_shots = 0
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT)
list_of_examples=[]
list_of_responses=[]
