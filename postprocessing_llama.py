from dotenv import dotenv_values
from datasets import load_dataset, Dataset
from utils.data_preprocessor import DataPreprocessor
from utils.test_data_processor import TestDataProcessor
from utils.generate_ft_adapters_list import generate_ft_adapters_list
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
from peft import PeftModel
from tqdm import tqdm

from config import postprocessing_params_llama as postprocessing
from log import llama7B_NoQuant_cl_lowlora as models_params


from datasets import Dataset
import os
import random
from transformers import AutoTokenizer
import warnings
import pandas as pd
import string
import pandas as pd
import string
from datasets import Dataset
from transformers import AutoTokenizer


class IOB_preprocessor():

    def __init__(self, model_checkpoint:str, tokenizer: AutoTokenizer, token_llama:str='', clen:bool=False) -> None:
        self.input_column= 'text'
        self.offset = None
        self.instruction_on_response_format = ''
        self.n_shots = None
        self.clen = clen
        #self.model_type = model_checkpoint.split('/')[1].lower().split('-')[0]
        self.model_type = 'qwen' if model_checkpoint.split('/')[0] == 'Qwen' else model_checkpoint.split('/')[1].lower().split('-')[0]
        if self.model_type == 'meta': self.model_type = 'llama3'
        # if self.model_type == 'zefiro':
        #     self.model_type  = 'mistral'
        if self.model_type not in ['mistral', 'llama', 'llama3', 'gemma', 'qwen', 'zefiro', 'phi', 'minerva']:
            raise ValueError("The model type must be either 'mistral', 'llama', 'llama3', 'gemma', 'zefiro', 'qwen', 'minerva' or 'phi'")

        print('MODEL TYPE:', self.model_type)
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, token = token_llama)
        else:
            self.tokenizer = tokenizer
        
        self.special_tokens_instruction_dict = {'mistral': {'user_start':'[INST]',
                                                            'user_end':'[/INST]',
                                                            'model_start':'',
                                                            'model_end':''},
                                                'llama': {'user_start':'[INST]',
                                                          'user_end':'[/INST]',
                                                          'model_start':'',
                                                          'model_end':''},
                                                'llama3': {'user_start':'<|start_header_id|>user<|end_header_id|>\n\n',
                                                          'user_end':'<|eot_id|>',
                                                          'model_start':'<|start_header_id|>assistant<|end_header_id|>\n\n',
                                                          'model_end':''},
                                                'gemma': {'user_start':'<start_of_turn>user',
                                                          'user_end':'<end_of_turn>',
                                                          'model_start':'<start_of_turn>model',
                                                          'model_end':'<end_of_turn>'},
                                                'qwen': {'user_start':'<|im_start|>user',
                                                          'user_end':'<|im_end|>',
                                                          'model_start':'<|im_start|>assistant',
                                                          'model_end':'<|im_end|>'},
                                                'zefiro': {'user_start':'<|user|>',
                                                           'user_end':'',# 'user_end':'</s>',
                                                           'model_start':'<|assistant|>',
                                                           'model_end':''},# 'model_end':'</s>'},
                                                'phi': {'user_start':'<|user|>',
                                                           'user_end':'<|end|>\n',
                                                           'model_start':'<|assistant|>',
                                                           'model_end':''},
                                                'minerva': {'user_start':'',
                                                            'user_end':'',
                                                           'model_start':'',
                                                           'model_end':''}}
        self.special_tokens_instruction = self.special_tokens_instruction_dict[self.model_type]

        self.one_shot_example = """{user_start} {instruction_on_response_format} <<<{example_query}>>> {user_end}{model_start} {example_response} {model_end}
"""
        self.one_shot_example_no_offset = """{user_start} {instruction_on_response_format} <<<{example_query}>>> {user_end}{model_start} {example_response} {model_end}
"""

        self.prompt_template = """{user_start} {instruction_on_response_format} <<{query}>>> {user_end}{model_start}"""

        self.prompt_template_no_offset = """{user_start} {instruction_on_response_format} <<{query}>>> {user_end}{model_start}"""



    def _base_prompt_input(self, input: str, instruction_on_response_format:str) -> str:
        """
        Format the input into a base prompt for the finetuning

        Args:
            input: the input text
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"

        Returns:
            the formatted base prompt
        """
        base_prompt = self.prompt_template_no_offset.format(
            instruction_on_response_format=instruction_on_response_format, 
            query=input,
            user_start=self.special_tokens_instruction['user_start'],
            user_end=self.special_tokens_instruction['user_end'],
            model_start=self.special_tokens_instruction['model_start'],
            model_end=self.special_tokens_instruction['model_end'])
            
        return base_prompt

    def _simplest_base_prompt_input(self, input: str) -> str:
        """
        Format the input and output into a prompt for the finetuning, in the simplest way possible, containing only the sentence and the response

        Args:
            input: the input text
            output: the output text

        Returns:
            the formatted prompt
        """
        base_prompt = self.special_tokens_instruction['user_start'] + input + self.special_tokens_instruction['user_end'] + self.special_tokens_instruction['model_start']
        return base_prompt

    def _format_prompt(self, input: str, instruction_on_response_format:str, simplest_prompt: bool, output:str='') -> str:
        """
        Format the input and output into a prompt for the finetuning

        Args:
            input: the input text
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            offset: whether to require the offset in the response
            output: the output text

        Returns:
            the formatted prompt
        """
        if output == '':
            raise ValueError("The output must be provided when generating prompts for the finetuning")
        
        if simplest_prompt:
            prompt_input = self._simplest_base_prompt_input(input)
        else:
            prompt_input = self._base_prompt_input(input, instruction_on_response_format)
        
        bos_token = self.tokenizer.bos_token
        eos_token = self.tokenizer.eos_token
        if self.model_type == 'qwen':
            bos_token = ''
            eos_token = ''
        # print(bos_token, prompt_input, output, self.special_tokens_instruction['model_end'], eos_token)
        prompt = bos_token + prompt_input + output + self.special_tokens_instruction['model_end'] + eos_token
                            
        return prompt


    def _format_entities_in_response(self, entities_list: [dict], offset: bool) -> str:
        """
        Format the response into a string

        Args:
            entities_list: the list of entities to format
            offset: whether to require the offset in the response
            
        Returns:
            the formatted response
        """
        formatted_response = '['
        if offset:
            for entity in entities_list:
                formatted_response = formatted_response + '{"entity": "' + entity['entity'] + f'", "offset": {entity["offsets"]}' + '}, '
        else:
            for entity in entities_list: 
                formatted_response = formatted_response + '{"entity": "' + entity['entity'] + '"}, '
        formatted_response = formatted_response[:-2]
        formatted_response = formatted_response + '] '
        return formatted_response
    
    def _apply_to_one_example(self, example, offset: bool, simplest_prompt: bool, instruction_on_response_format:str) -> dict:
        """
        Apply the data preprocessing to one example

        Args:
            example: the example (data row) to preprocess
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            offset: whether to require the offset in the response
            simplest_prompt: whether to generate the prompt or just concatenate the sentence and the response

        Returns:
            the preprocessed example
        """
        output = self._format_entities_in_response(entities_list=example['entities'], offset=offset)
        prompt = self._format_prompt(input=example[self.input_column], 
                                     simplest_prompt=simplest_prompt,
                                     instruction_on_response_format=instruction_on_response_format,
                                     output=output)
        example['prompt'] = prompt
        return example
    

    def _only_clent_from_enities(self, example):
        example['entities'] = [entity for entity in example['entities'] if entity['type'] == 'CLINENTITY']
        return example
    
    
    def apply(self, instruction_on_response_format:str, offset: bool,  simplest_prompt:bool, num_proc: int=1): # -> Dataset:
        """
        Apply the data preprocessing to one split/layer if the dataset. It formats the prompt in the right shape, processing the entities.

        Args:
            data: the dataset to preprocess
            instruction_on_response_format: the instruction on the response format to be given to the model. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            n_shots: the number of examples to provide as few shot prompting   
            offset: whether to require the offset in the response  
            num_proc: the number of processes to use for the parallel processing

        Returns:
            the preprocessed split/layer
        """
        self.data = self.data.map(lambda example:  self._apply_to_one_example(example=example, 
                                                                    simplest_prompt=simplest_prompt,
                                                                    instruction_on_response_format = instruction_on_response_format, 
                                                                    offset = offset), 
                        num_proc=num_proc) #batched=True)
        self.offset = offset
        self.instruction_on_response_format = instruction_on_response_format
        self.simplest_prompt = simplest_prompt
        # return data



class DataPreprocessor(IOB_preprocessor):

    def __init__(self, model_checkpoint:str, tokenizer: AutoTokenizer, token_llama:str='', clen:bool=False) -> None:
        super().__init__( model_checkpoint, tokenizer, token_llama, clen)
        self.input_column = 'sentence'

    def _apply_to_one_example(self, example, offset: bool, simplest_prompt: bool, instruction_on_response_format:str, ) -> dict:
        """
        Apply the data preprocessing to one example

        Args:
            example: the example (data row) to preprocess
            instruction_on_response_format: the instruction on the response format. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            offset: whether to require the offset in the response
            simplest_prompt: whether to generate the prompt or just concatenate the sentence and the response

        Returns:
            the preprocessed example
        """
        output = self._format_entities_in_response(entities_list=example['entities'], offset=offset)
        prompt = self._format_prompt(input=example[self.input_column], 
                                     simplest_prompt=simplest_prompt,
                                     instruction_on_response_format=instruction_on_response_format,
                                     output=output)
        example['prompt'] = prompt
        return example
    
    def _format_entities_in_response(self, entities_list: [dict], offset: bool) -> str:
        """
        Format the response into a string

        Args:
            entities_list: the list of entities to format
            offset: whether to require the offset in the response
            
        Returns:
            the formatted response
        """
        formatted_response = '['
        if offset:
            for entity in entities_list:
                formatted_response = formatted_response + '{"entity": "' + entity['text'] + '"}, '
        else:
            for entity in entities_list: 
                formatted_response = formatted_response + '{"entity": "' + entity['text'] + '"}, '
        #print(formatted_response )
        if formatted_response == '[':
            formatted_response = '[{"entity": ""}]'
        else:
            formatted_response = formatted_response[:-2]
            formatted_response = formatted_response + '] '
        #print(formatted_response, '\n')
        return formatted_response

    
    def apply(self, data: Dataset, instruction_on_response_format:str, offset: bool,  simplest_prompt:bool, num_proc: int=1) -> Dataset:
        """
        Apply the data preprocessing to one split/layer if the dataset. It formats the prompt in the right shape, processing the entities.

        Args:
            data: the dataset to preprocess
            instruction_on_response_format: the instruction on the response format to be given to the model. E.g. "The response must be a list of dictionaries, where each dictionary contains the keys 'text' and 'offset'"
            n_shots: the number of examples to provide as few shot prompting   
            offset: whether to require the offset in the response  
            num_proc: the number of processes to use for the parallel processing

        Returns:
            the preprocessed split/layer
        """
        if self.clen:
            data = data.map(self._only_clent_from_enities, num_proc=1)
        data = data.map(lambda example:  self._apply_to_one_example(example=example, 
                                                                    simplest_prompt=simplest_prompt,
                                                                    instruction_on_response_format = instruction_on_response_format, 
                                                                    offset = offset), 
                        num_proc=num_proc) #batched=True)
        self.offset = offset
        self.instruction_on_response_format = instruction_on_response_format
        self.simplest_prompt = simplest_prompt
        return data


    
    def preprocess_data_one_layer(self, hf_dataset: Dataset, instruction_on_response_format:str='', offset:bool=False, simplest_prompt:bool=False) -> Dataset:
        """
        Preprocess one layer/split of the dataset the trasformations defined in self.apply()

        Args:
            hf_dataset: one layer/split of the dataset to preprocess

        Returns:
            the preprocessed dataset
        """
        if not simplest_prompt and instruction_on_response_format == '':
            raise ValueError("The instruction_on_response_format must be provided when not using the simplest_prompt")
            
        hf_dataset = self.apply(data=hf_dataset, 
                                instruction_on_response_format=instruction_on_response_format, 
                                offset=offset,
                                simplest_prompt=simplest_prompt)
        return hf_dataset
    
    def split_layer_into_train_val_test_(self, dataset: Dataset, split_name: str, test_subset_of_validation: bool=False) -> (Dataset, Dataset):
        """
        Split the layer into train, validation and test sets, according to the split defined at https://github.com/hltfbk/E3C-Corpus/tree/main/documentation

        Args:
            dataset: the dataset to split. Must be a split of the original Hugging Face dataset
            split_name: the name of the layer
            test_subset_of_validation: wether the test set is a subset of the validation set. Set this to True if you want to use the test set as a way of checking on the training throw wandb
                                to mantain the diviosn it train-test of the original repository. Default is False.
        
        Returns:
            the train and test sets
        """
        mapping = {'en.layer1': 'train_labels_en.txt', 
                'es.layer1': 'train_labels_es.txt',
                'eu.layer1': 'train_labels_eu.txt',
                'it.layer1': 'train_labels_it.txt',
                'fr.layer1': 'train_labels_fr.txt',}
        labels_path = mapping[split_name]
        with open(os.path.join('data', labels_path), 'r') as file:
            file_content = file.read()
        labels = file_content.split(", ")
        labels = [label[1:-1] for label in labels]
        idxs_train = [idx for idx, x in enumerate(dataset['original_id']) if x in labels]
        idxs_val = [idx for idx, x in enumerate(dataset['original_id']) if x not in labels]
        random.seed(42)
        idxs_test = random.sample(idxs_val, int(len(idxs_val) * 0.2))
        train_data = dataset.select(idxs_train)
        test_data = dataset.select(idxs_test)
        if test_subset_of_validation:
            val_data = dataset.select(idxs_val)
        else:
            idxs_val = [idx for idx in idxs_val if idx not in idxs_test]
            val_data = dataset.select(idxs_val)

        if self.offset:
            prompt_template = self.prompt_template
        else:
            prompt_template = self.prompt_template_no_offset
        
        def remove_answer_from_prompt(example):
            prompt_no_answ = prompt_template.format(instruction_on_response_format=self.instruction_on_response_format, query=example[self.input_column],
                                                    user_start=self.special_tokens_instruction['user_start'],
                                                    user_end=self.special_tokens_instruction['user_end'],
                                                    model_start=self.special_tokens_instruction['model_start'],
                                                    model_end=self.special_tokens_instruction['model_end'])
            example['prompt_with_answer'] = example['prompt']
            example['prompt'] = prompt_no_answ
            return example

        test_data = test_data.map(remove_answer_from_prompt, batched=False)

        return train_data, val_data, test_data




adapters_list = generate_ft_adapters_list("llama7B_NoQuant_cl_lowlora", simplest_prompt=models_params.simplest_prompt)
print(adapters_list)
HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
LLAMA_TOKEN = dotenv_values(".env.base")['LLAMA_TOKEN']

max_new_tokens_factor_list = postprocessing.max_new_tokens_factor_list
n_shots_inference_list = postprocessing.n_shots_inference_list
layer = models_params.TRAIN_LAYER
language = layer.split('.')[0]

dataset = load_dataset("ferrazzipietro/e3c-sentences", token=HF_TOKEN)
dataset = dataset[layer]
tokenizer = AutoTokenizer.from_pretrained(models_params.BASE_MODEL_CHECKPOINT, add_eos_token=False,
                                         token=LLAMA_TOKEN)
preprocessor = DataPreprocessor(model_checkpoint=models_params.BASE_MODEL_CHECKPOINT, 
                                tokenizer = tokenizer, clen=models_params.clent)
dataset = preprocessor.preprocess_data_one_layer(dataset,
                                                 models_params.instruction_on_response_format)
_, val_data, _ = preprocessor.split_layer_into_train_val_test_(dataset, layer)

print("val_data:", val_data[0:15])


from transformers import StoppingCriteria
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [518, 29914, 25580, 29962]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        eos_cond = 2 in input_ids[:,-1].tolist()
        return (self.eos_sequence in last_ids) or eos_cond



print("val_data:", val_data[0:3])



from transformers import StoppingCriteria
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [518, 29914, 25580, 29962]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        eos_cond = 2 in input_ids[:,-1].tolist()
        return (self.eos_sequence in last_ids) or eos_cond



print("val_data:", val_data[0:3])

for max_new_tokens_factor in max_new_tokens_factor_list:
    for n_shots_inference in n_shots_inference_list:
        for adapters in tqdm(adapters_list, desc="adapters_list"):
            if adapters.endswith("0.0008"):
                continue
            print("PROCESSING:", adapters, "n_shots_inference:", n_shots_inference, "max_new_tokens_factor:", max_new_tokens_factor)
            if not models_params.quantization:
                print("NO QUANTIZATION")
                base_model = AutoModelForCausalLM.from_pretrained(
                    models_params.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
                    return_dict=True,  
                    torch_dtype=postprocessing.torch_dtype,
                    device_map= "auto",
                    token=LLAMA_TOKEN)    
            else:
                print("QUANTIZATION")
                load_in_8bit = not models_params.load_in_4bit[0]
                load_in_4bit = models_params.load_in_4bit[0]
                load_in_8bit = not load_in_4bit
                bnb_4bit_use_double_quant = models_params.bnb_4bit_use_double_quant
                bnb_4bit_quant_type = models_params.bnb_4bit_quant_type[0]
                bnb_4bit_compute_dtype = models_params.bnb_4bit_compute_dtype[0]
                llm_int8_threshold = models_params.llm_int8_threshold[0]
                # llm_int8_has_fp16_weight = models_params.llm_int8_has_fp16_weight AVOID IT AT INFERENCE TIME!
                # llm_int8_skip_modules = models_params.llm_int8_skip_modules AVOID IT AT INFERENCE TIME!

                bnb_config = BitsAndBytesConfig(
                            load_in_4bit=load_in_4bit,
                            load_in_8bit=load_in_8bit,
                            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                            bnb_4bit_quant_type=bnb_4bit_quant_type,
                            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                            llm_int8_threshold=llm_int8_threshold ,
                            # llm_int8_has_fp16_weight =True #,AVOID IT AT INFERENCE TIME!
                            # llm_int8_skip_modules=llm_int8_skip_modules AVOID IT AT INFERENCE TIME!
                            )
                base_model = AutoModelForCausalLM.from_pretrained(
                    models_params.BASE_MODEL_CHECKPOINT, low_cpu_mem_usage=True,
                    quantization_config = bnb_config,
                    return_dict=True,  
                    device_map= "auto",
                    token=LLAMA_TOKEN)
            merged_model = PeftModel.from_pretrained(base_model, 
                                                     adapters, 
                                                     token=HF_TOKEN, 
                                                     device_map='auto',
                                                     is_trainable = False)
            tokenizer = AutoTokenizer.from_pretrained(models_params.BASE_MODEL_CHECKPOINT, 
                                                      add_eos_token=False,
                                                      token=LLAMA_TOKEN)
            tokenizer.pad_token = tokenizer.eos_token# "<pad>" #tokenizer.eos_token
            tokenizer.padding_side = "left"
#            tokenizer = AutoTokenizer.from_pretrained(models_params.BASE_MODEL_CHECKPOINT, add_eos_token=True, token=LLAMA_TOKEN)
#            tokenizer.add_special_tokens({"pad_token":"<pad>"})
#            merged_model.resize_token_embeddings(len(tokenizer))
#            print('tokenizer.pad_token_id:', tokenizer.pad_token_id)
#            merged_model.config.pad_token_id = tokenizer.pad_token_id

            postprocessor = TestDataProcessor(test_data=val_data, 
                                              preprocessor=preprocessor, 
                                              n_shots_inference=n_shots_inference, 
                                              language=language, 
                                              tokenizer=tokenizer)
            postprocessor.add_inference_prompt_column(simplest_prompt=False)

            # tmp = []
            # for example in postprocessor.test_data:
            #     tmp.append(example)
            # import pandas as pd
            # tmp = pd.DataFrame(tmp)
            # tmp = tmp.iloc[tmp['inference_prompt'].str.len().argsort()]
            # postprocessor.test_data = Dataset.from_pandas(tmp)

            postprocessor.add_ground_truth_column()
            #try:
            postprocessor.add_responses_column(model=merged_model, 
                                            tokenizer=tokenizer, 
                                            batch_size=postprocessing.batch_size, 
                                            max_new_tokens_factor=max_new_tokens_factor,
                                            stopping_criteria = [EosListStoppingCriteria()],
                                            temperature=postprocessing.temperature)
            postprocessor.test_data.to_csv(f"{postprocessing.save_directory}maxNewTokensFactor{max_new_tokens_factor}_nShotsInference{n_shots_inference}_{adapters.split('/')[1]}.csv", index=False)
            # except RuntimeError as e:
                # print("ERROR IN PROCESSING: ", e, adapters)
                # print(e.message)
            del merged_model
            if models_params.quantization: del base_model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()



