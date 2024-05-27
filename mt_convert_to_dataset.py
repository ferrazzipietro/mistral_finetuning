from utils.data_preprocessor import  MT_preprocessor

import pandas as pd
df = pd.read_csv('data/MTSamples/test/1.bio',delim_whitespace=True, header=None, names=['Word', 'Tag'])
file_path = 'data/MTSamples/test/22.bio'

from dotenv import dotenv_values
HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']

mt_preprocessor = MT_preprocessor( "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", token_llama=HF_TOKEN)
mt_preprocessor.preprocess('data/MTSamples/test')
mt_preprocessor.preprocess('data/MTSamples/train')
dataset = mt_preprocessor.push_dataset_to_hub('ferrazzipietro/mt_samples_problems', HF_TOKEN, private=True)
