from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import config
from dotenv import dotenv_values
from utils.data_preprocessor import DataPreprocessor

HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

print('TEST')

base_model_reload = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
    return_dict=True,  load_in_4bit=True, # torch_dtype=torch.float16,
    device_map= "auto")

adp = config.ADAPTERS_CHECKPOINT
merged_model = PeftModel.from_pretrained(base_model_reload, adp, token=HF_TOKEN, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_CHECKPOINT, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


print(type(merged_model))

def get_completion_merged(prompt: str, merged_model, tokenizer) -> str:
  device = "cuda"
  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
  model_inputs = encodeds.to(device)
  generated_ids = merged_model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])


first_input = 'We present a case of a 32-year-old woman with a history of gradual enlargement of the anterior neck.'
second_input = 'Patient information: a 9-month-old boy presented to the emergency room with a 3-day history of refusal to bear weight on the right lower extremity and febrile peaks of up to 38.5°C for 24 hours.'
first_response = '[{"entity": "present", "offset": [3, 10]}, {"entity": "history", "offset": [48, 55]}, {"entity": "enlargement", "offset": [67, 78]}]'
second_response = '[{"entity": "presented", "offset": [39, 48]}, {"entity": "refusal", "offset": [95, 102]}, {"entity": "bear", "offset": [106, 110]}, {"entity": "peaks", "offset": [159, 164]}]'


query_ita = """Per un miglior inquadramento diagnostico
 abbiamo eseguito rx torace, rx t.d. prime vie, egdS, eco addome e rMn, che evidenziavano la presenza
 di massivo reflusso gastro-esofageo, megaesofago con cardias dilatato, microgastria, atresia incompleta del corpo dello stomaco, ernia
 diaframmatica laterale sinistra (colon)."""

query_eng = "A 46-year-old man with hypertension and dyslipidemia diagnosed 4-months before, as well as new-onset diabetes mellitus unveiled 1-month earlier, was referred to emergency department for hypokalemia"

preprocessor = DataPreprocessor()
prompt = preprocessor._format_prompt(task='inference', 
                                     input=query_eng, 
                                     instruction_on_response_format='Return a json format', 
                                     offset=True, 
                                     tokenizer=tokenizer, 
                                     output='', 
                                     n_shots=2, 
                                     list_of_examples=[first_input, second_input], 
                                     list_of_responses=[first_response, second_response])

result_NO_ft = get_completion_merged(prompt=query_eng, 
                               merged_model=base_model_reload, 
                               tokenizer=tokenizer)

result_ft = get_completion_merged(prompt=query_eng, 
                               merged_model=merged_model, 
                               tokenizer=tokenizer)
print(f"result_ft:\n{result_ft}\n\n\nresult_NO_ft:\n{result_NO_ft}")


# offset: [23, 35] text: hypertension ||| offset: [40, 52] text: dyslipidemia ||| offset: [53, 62] text: diagnosed ||| offset: [110, 118] text: mellitus ||| offset: [149, 157] text: referred ||| offset: [186, 197] text: hypokalemia