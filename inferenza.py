from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import config
from dotenv import dotenv_values

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

print('TEST')

base_model_reload = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
    return_dict=True,torch_dtype=torch.float16,
    device_map= "auto")

adp = config.ADAPTERS_CHECKPOINT
merged_model = PeftModel.from_pretrained(base_model_reload, adp, token=HF_TOKEN, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_CHECKPOINT, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


print(type(merged_model))

def get_completion_merged(query: str, merged_model, tokenizer) -> str:
  device = "cuda:0"
  # first_response = "offset: [3, 10] text: present ||| offset: [48, 55] text: history ||| offset: [67, 78] text: enlargement"
  first_response = '[{"entity": "present", "offset": [3, 10]}, {"entity": "history", "offset": [48, 55]}, {"entity": "enlargement", "offset": [67, 78]}]'
  # first_response = '{ "present": [3, 10], "history":[48, 55], "enlargement":[67, 78] }'
  second_response = '[ {"entity": "presented", "offset": [39, 48]}, {"entity": "refusal", "offset": [95, 102]}, {"entity": "bear", "offset": [106, 110]}, {"entity": "peaks", "offset": [159, 164]}]'

  #   Return the result in a json format where the entity name is the key and the offset is the value.
  prompt_template = """
  <s>
  [INST]
  Extract the entities contained in the text and the offset, i.e. the position of that entity in the string. Extract only entities contained in the text.
  Return the result in a json format.
  Text: <<<We present a case of a 32-year-old woman with a history of gradual enlargement of the anterior neck.>>>
  [/INST]
  {first_response}
  [INST]
  Extract the entities contained in the text and the offset, i.e. the position of that entity in the string. Extract only entities contained in the text.
  Return the result in a json format.
  Text: <<<Patient information: a 9-month-old boy presented to the emergency room with a 3-day history of refusal to bear weight on the right lower extremity and febrile peaks of up to 38.5°C for 24 hours.>>>
  [/INST]
  {second_response}
  [INST]

  Extract the entities contained in the text and the offset, i.e. the position of that entity in the string. Extract only entities contained in the text.
  Return the result in a json format.

  Text: <<{query}>>>
  [/INST]

  """
  prompt = prompt_template.format(query=query, first_response=first_response, second_response=second_response)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

  model_inputs = encodeds.to(device)

  generated_ids = merged_model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])

result = get_completion_merged(query="A 46-year-old man with hypertension and dyslipidemia diagnosed 4-months before, as well as new-onset diabetes mellitus unveiled 1-month earlier, was referred to emergency department for hypokalemia", 
                               merged_model=base_model_reload, tokenizer=tokenizer)
print(result)


# offset: [23, 35] text: hypertension ||| offset: [40, 52] text: dyslipidemia ||| offset: [53, 62] text: diagnosed ||| offset: [110, 118] text: mellitus ||| offset: [149, 157] text: referred ||| offset: [186, 197] text: hypokalemia