from datasets import load_dataset
import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import dotenv_values
from utils.process_split import Process_split 


HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
hf_e3c = load_dataset("bio-datasets/e3c")

data_ft = pd.DataFrame(columns=['input', 'output', 'layer'])
splits_dict = {}
splits = ['en.layer1', 'en.layer2', 'en.layer2.validation', 'en.layer3',
          'es.layer1', 'es.layer2', 'es.layer2.validation', 'es.layer3',
          'eu.layer1', 'eu.layer2', 'eu.layer2.validation', 'eu.layer3',
          'it.layer1', 'it.layer2', 'it.layer2.validation', 'it.layer3',
          'fr.layer1', 'fr.layer2', 'fr.layer2.validation', 'fr.layer3']

for split_name in splits:
    processed_split = Process_split.apply(hf_e3c, split_name, enitites_separator_in_output="|||")
    data_ft = Dataset.from_pandas(processed_split)
    splits_dict[split_name] = data_ft

ddict = DatasetDict(splits_dict)
ddict.push_to_hub("ferrazzipietro/e3c_finetuning", token=HF_TOKEN)
    
data_ft = Dataset.from_pandas(data_ft)
data_ft.push_to_hub("ferrazzipietro/e3c_finetuning", token=HF_TOKEN)