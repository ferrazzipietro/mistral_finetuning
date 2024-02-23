from transformers import AutoTokenizer
from .config import BASE_MODEL_CHECKPOINT

offset=False
instruction_on_response_format='Return the result in a json format: [{"entity":"entity_name"}].'# 'Return the result in a json format.'
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT)
