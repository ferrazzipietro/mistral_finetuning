from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import config
from dotenv import dotenv_values

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

print('TEST')

base_model_reload = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1", low_cpu_mem_usage=True,
    return_dict=True,torch_dtype=torch.float16,
    device_map= "auto")

adp = "ferrazzipietro/Mistral-7B-Instruct-v0.2_adapters_en.layer1"
merged_model = PeftModel.from_pretrained(base_model_reload, adp, token=HF_TOKEN, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"
# tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

print(type(merged_model))



def get_completion_ita(query: str, merged_model, tokenizer) -> str:
  device = "cuda"
  first_response = '[{"entity": "reflusso gastro-esofageo"}, { "entity": "ritardo di crescita"}, { "entity": "deficit protidocalorico"}]'

  prompt_template_short = """
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

  Text: <<{query}>>>
  [/INST]

  """

  prompt_short = prompt_template_short.format(query=query, first_response=first_response)
  encodeds = tokenizer(prompt_short, return_tensors="pt", add_special_tokens=False)

  model_inputs = encodeds.to(device)

  generated_ids = merged_model.generate(**model_inputs, max_new_tokens=750, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])


def get_completion_eng(query: str, merged_model, tokenizer) -> str:
  device = "cuda"
  # first_response_offset = '[{"entity": "present", "offset": [3, 10]}, {"entity": "history", "offset": [48, 55]}, {"entity": "enlargement", "offset": [67, 78]}]'
  # second_response_offset = '[{"entity": "presented", "offset": [39, 48]}, {"entity": "refusal", "offset": [95, 102]}, {"entity": "bear", "offset": [106, 110]}, {"entity": "peaks", "offset": [159, 164]}]'

  first_response = '[{"entity": "present"}, {"entity": "history"}, {"entity": "enlargement"}]'
  second_response = '[{"entity": "presented"}, {"entity": "refusal"}, {"entity": "bear"}, {"entity": "peaks"}]'
  # Return the result in a json format where the entity name is the key and the offset is the value.
  prompt_template = """
  <s>
  [INST]
  Extract the entities contained in the text. Extract only entities contained in the text.
  Return the result in a json format.
  Text: <<<We present a case of a 32-year-old woman with a history of gradual enlargement of the anterior neck.>>>
  [/INST]
  {first_response}
  [INST]
  Extract the entities contained in the text. Extract only entities contained in the text.
  Return the result in a json format.
  Text: <<<Patient information: a 9-month-old boy presented to the emergency room with a 3-day history of refusal to bear weight on the right lower extremity and febrile peaks of up to 38.5°C for 24 hours.>>>
  [/INST]
  {second_response}
  [INST]

  Extract the entities contained in the text. Extract only entities contained in the text.
  Return the result in a json format.

  Text: <<{query}>>>
  [/INST]

  """

  prompt_template_short = """
  <s> [INST] Extract the entities contained in the text. Extract only entities contained in the text.
  Return the result in a json format.
  Text: <<<We present a case of a 32-year-old woman with a history of gradual enlargement of the anterior neck.>>>
  [/INST]
  {first_response}
  
  [INST] Extract the entities contained in the text. Extract only entities contained in the text.
  Return the result in a json format.

  Text: <<{query}>>>
  [/INST]
  """
  prompt = prompt_template.format(query=query, first_response=first_response, second_response=second_response)
  prompt_short = prompt_template_short.format(query=query, first_response=first_response)
  encodeds = tokenizer(prompt_short, return_tensors="pt", add_special_tokens=False)

  model_inputs = encodeds.to(device)

  generated_ids = merged_model.generate(**model_inputs, max_new_tokens=750, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])

query_eng = "A 46-year-old man with hypertension and dyslipidemia diagnosed 4-months before, as well as new-onset diabetes mellitus unveiled 1-month earlier, was referred to emergency department for hypokalemia"
# result_NO_ft = get_completion_eng(query=query_eng, 
#                                merged_model=base_model_reload, tokenizer=tokenizer)
# result_ft = get_completion_eng(query=query_eng, 
#                                merged_model=merged_model, tokenizer=tokenizer)



query_ita = """Per un miglior inquadramento diagnostico
 abbiamo eseguito rx torace, rx t.d. prime vie, egdS, eco addome e rMn, che evidenziavano la presenza
 di massivo reflusso gastro-esofageo, megaesofago con cardias dilatato, microgastria, atresia incompleta del corpo dello stomaco, ernia
 diaframmatica laterale sinistra (colon)."""

result_NO_ft = get_completion_ita(query=query_ita, 
                               merged_model=base_model_reload, tokenizer=tokenizer)
result_ft = get_completion_ita(query=query_ita, 
                               merged_model=merged_model, tokenizer=tokenizer)


print(f"result NO FINETUNING:\n {result_NO_ft}\n\n\nResult FINETUNING:\n {result_ft}\n\n expected: hypertension, dyslipidemia, diagnosed, mellitus, referred, hypokalemia")


# offset: [23, 35] text: hypertension ||| offset: [40, 52] text: dyslipidemia ||| offset: [53, 62] text: diagnosed ||| offset: [110, 118] text: mellitus ||| offset: [149, 157] text: referred ||| offset: [186, 197] text: hypokalemia