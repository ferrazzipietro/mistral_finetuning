import pandas as pd
import re

def refine_output_df(res):

    def clean_quantization(example):
        if example['quantization'] == 'NoQuantbit':
            example['quantization'] = 'NoQuant'
        elif example['quantization'] == '4bits':
            example['quantization'] = '4bit'
        elif example['quantization'] == 'noInstr':
            example['quantization'] = example['fine_tuning']
            example['fine_tuning'] = 'unsure'
        return example
    
    res = res.drop(['training_params_string', 'similar_is_equal_threshold', 'Unnamed: 0', 'similar_is_equal'], axis = 1)
    res = res.apply(lambda x: clean_quantization(x), axis = 1)
    return res

def extract_params_from_file_name(df: pd.DataFrame):

    def adjust_model_size_format(model_size_str):
        try:
            int(model_size_str)
        except ValueError:
            model_size_str = model_size_str.replace('b', '').replace('B', '')
        return model_size_str
    
    def extract_if_noInstruct(string):
        if 'noInstr' in string:
            return 'notInstructed'
        return 'instructed'


    df['model_type'] = df['file'].apply(lambda x: str(x.split('/')[1]))
    df['instructed'] = df['file'].apply(lambda x: extract_if_noInstruct(x))
    df['model_configurations'] = df['file'].apply(lambda x: str(x.split('/')[2]))
    print(df['model_configurations'][0])
    if df['model_type'][0] in ['mistral', 'zefiro', 'phi3']:
        df['model_size'] = '7'
        if df['model_type'][0] == 'phi3':
            df['model_size'] = '3'
        df['quantization'] = df['model_configurations'].apply(lambda x: str(x.split('_')[0]))
        try:
            df['fine_tuning'] = df['model_configurations'].apply(lambda x: str(x.split('_')[1]))
        except IndexError:
            df['fine_tuning'] = 'FT'
    else:
        df['model_size'] = df['model_configurations'].apply(lambda x: adjust_model_size_format(x.split('_')[0]))
        df['quantization'] = df['model_configurations'].apply(lambda x: str(x.split('_')[1]))
        df['fine_tuning'] = df['model_configurations'].apply(lambda x: str(x.split('_')[2]))

    df['maxNewTokensFactor'] = df['file'].apply(lambda x: re.search(r'maxNewTokensFactor(\d+)', x).group(1))
    df['nShotsInference'] = df['file'].apply(lambda x: re.search(r'nShotsInference(\d+)', x).group(1))
    #
    # df['layer'] = df['file'].apply(lambda x: re.search(r'adapters_(\s+)', x).group(1))
    df['model'] = df['file'].apply(lambda x: str(x.split('_adapters_')[0].split('nShotsInference')[1][2:].replace('.csv', '')))
    if df['fine_tuning'][0] == 'FT':
        df['training_params_string'] = df['file'].apply(lambda x: x.split('adapters_')[1])
        # df['nbit'] = df['training_params_string'].apply(lambda x: x.split('_')[1])
        df['bnb_4bit_compute_dtype'] = df['training_params_string'].apply(lambda x: x.split('_')[2])
        df['r'] = df['training_params_string'].apply(lambda x: int(x.split('_')[3]))
        df['lora_alpha'] = df['training_params_string'].apply(lambda x: int(x.split('_')[4]))
        df['lora_dropout'] = df['training_params_string'].apply(lambda x: float(x.split('_')[5]))
        df['gradient_accumulation_steps'] = df['training_params_string'].apply(lambda x: int(x.split('_')[6]))
        df['learning_rate'] = df['training_params_string'].apply(lambda x: x.split('_')[7].replace('.csv', ''))
    elif df['fine_tuning'][0] in ['base', 'NoFT']:
        df['training_params_string'] = None
    return df