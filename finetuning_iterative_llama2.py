# https://colab.research.google.com/github/adithya-s-k/LLM-Alchemy-Chamber/blob/main/LLMs/Mistral-7b/Mistral_Colab_Finetune_ipynb_Colab_Final.ipynb?source=post_page-----0f39647b20fe--------------------------------#scrollTo=acCr5AZ0831z




import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from trl import SFTTrainer
from dotenv import dotenv_values
import wandb
from utils.data_preprocessor import DataPreprocessor
import datetime
import gc

from config.finetuning_llama2 import training_params, lora_params, model_loading_params, config, preprocessing_params



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
        print(formatted_response )
        if formatted_response == '[':
            formatted_response = '[{"entity": ""}]'
        else:
            formatted_response = formatted_response[:-2]
            formatted_response = formatted_response + '] '
        print(formatted_response, '\n')
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



HF_TOKEN = dotenv_values(".env.base")['HF_TOKEN']
WANDB_KEY = dotenv_values(".env.base")['WANDB_KEY']
LLAMA_TOKEN = dotenv_values(".env.base")['LLAMA_TOKEN']



def main(ADAPTERS_CHECKPOINT,
         load_in_4bit, bnb_4bit_quant_type, bnb_4bit_compute_dtype, llm_int8_threshold,
         r, lora_alpha, lora_dropout,
         gradient_accumulation_steps,learning_rate):
  
  # Monitering the LLM
  wandb.login(key = WANDB_KEY)
  run = wandb.init(project=config.WANDB_PROJECT_NAME, job_type="training", anonymous="allow",
                  name=ADAPTERS_CHECKPOINT.split('/')[1],
                  config={'model': config.BASE_MODEL_CHECKPOINT, 
                          'dataset': config.DATASET_CHEKPOINT, 
                          'layer': config.TRAIN_LAYER,
                          'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
  
  find_llm_int8_skip_modules = False
  if find_llm_int8_skip_modules:
    def find_all_linear_names(model):
      cls = bnb.nn.Linear4bit if load_in_4bit else (bnb.nn.Linear8bitLt if load_in_8bit 
      else torch.nn.Linear)
      lora_module_names = set()
      for name, module in model.named_modules():
        if isinstance(module, cls):
          names = name.split('.')
          lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
          lora_module_names.remove('lm_head')
      return list(lora_module_names)
    linear_modules = find_all_linear_names(model)
    llm_int8_skip_modules = []
    if load_in_8bit:
      llm_int8_skip_modules = linear_modules
  else:
    llm_int8_skip_modules = []
    if load_in_8bit:
      llm_int8_skip_modules = model_loading_params.llm_int8_skip_modules
  
  bnb_config = BitsAndBytesConfig(
      load_in_4bit= load_in_4bit,
      load_in_8bit = load_in_8bit,

      bnb_4bit_quant_type= bnb_4bit_quant_type,
      bnb_4bit_compute_dtype= bnb_4bit_compute_dtype,
      bnb_4bit_use_double_quant= model_loading_params.bnb_4bit_use_double_quant,

      llm_int8_threshold= llm_int8_threshold,
      llm_int8_skip_modules= llm_int8_skip_modules,
      # llm_int8_has_fp16_weight= model_loading_params.llm_int8_has_fp16_weight # Had to comment this to run llama 7B in 8 bit. There are numerical issues with fp16. I will instead use the default float16
  )

  if not model_loading_params.quantization:
    model = AutoModelForCausalLM.from_pretrained(
      config.BASE_MODEL_CHECKPOINT,
      device_map="auto",
      token=LLAMA_TOKEN,
      torch_dtype=model_loading_params.torch_dtype,
      )
    model.gradient_checkpointing_enable() # Activates gradient checkpointing for the current model.
    model.config.use_cache = False
  else:
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_CHECKPOINT,
        quantization_config=bnb_config,
        device_map="auto",
        token=LLAMA_TOKEN
    )
    """
    prepare_model_for_kbit_training wraps the entire protocol for preparing a model before running a training. 
            This includes:  1- Cast the layernorm in fp32 
                            2- making output embedding layer require gradient (needed as you are going to train (finetune) the model)
                            3- upcasting the model's head to fp32 for numerical stability
    """
  
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable() # Activates gradient checkpointing for the current model.
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
  #Adding the adapters in the layers

  tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_CHECKPOINT, add_eos_token=False,
                                            token = LLAMA_TOKEN) #, cache_dir='/data/disk1/share/pferrazzi/.cache')
  tokenizer.pad_token = '<pad>'#tokenizer.eos_token
  tokenizer.padding_side = 'right'

  # tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_CHECKPOINT, add_eos_token=True, token=LLAMA_TOKEN)
  # tokenizer.add_special_tokens({"pad_token":"<pad>"})
  # model.resize_token_embeddings(len(tokenizer))
  # print('tokenizer.pad_token_id:', tokenizer.pad_token_id)
  # model.config.pad_token_id = tokenizer.pad_token_id
  # # model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size, model.config.padding_idx)
  # # tokenizer.pad_token = tokenizer.unk_token
  # tokenizer.padding_side = 'right'

  preprocessor = DataPreprocessor(config.BASE_MODEL_CHECKPOINT, 
                                  tokenizer, clen=preprocessing_params.clent)
  dataset = load_dataset(config.DATASET_CHEKPOINT) #download_mode="force_redownload"
  dataset = dataset[config.TRAIN_LAYER]
  dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
  dataset = preprocessor.preprocess_data_one_layer(dataset, 
                                                   instruction_on_response_format=preprocessing_params.instruction_on_response_format,
                                                   simplest_prompt=preprocessing_params.simplest_prompt)
  dataset = dataset.map(lambda samples: tokenizer(samples[training_params.dataset_text_field]), batched=True)
  train_data, val_data, test_data = preprocessor.split_layer_into_train_val_test_(dataset, config.TRAIN_LAYER)
  print('train: ', 
        train_data[0])

  lora_config = LoraConfig(
          r=r,
          lora_alpha=lora_alpha,
          lora_dropout=lora_dropout,
          bias=lora_params.bias,
          task_type=lora_params.task_type,
          target_modules=lora_params.target_modules # lora_params.target_modules
          )
  model = get_peft_model(model, lora_config)

  torch.cuda.empty_cache()

  #Hyperparamter
  
  training_arguments = TrainingArguments(
      output_dir= "./training_output",
      push_to_hub=True,
      hub_model_id=config.FT_MODEL_CHECKPOINT,
      hub_token=HF_TOKEN,
      hub_private_repo=True,
      num_train_epochs= training_params.num_train_epochs,
      per_device_train_batch_size= training_params.per_device_train_batch_size,
      per_device_eval_batch_size= training_params.per_device_train_batch_size,
      gradient_accumulation_steps= gradient_accumulation_steps,
      optim=  training_params.optim,
      learning_rate= learning_rate,
      weight_decay= training_params.weight_decay,
      fp16= training_params.fp16,
      bf16= training_params.bf16,
      max_grad_norm= training_params.max_grad_norm,
      max_steps= training_params.max_steps,
      warmup_ratio= training_params.warmup_ratio,
      group_by_length= training_params.group_by_length,
      lr_scheduler_type= training_params.lr_scheduler_type,
      report_to="wandb",

      logging_steps= training_params.logging_steps, 
      logging_strategy= training_params.logging_strategy, 
      evaluation_strategy= training_params.evaluation_strategy, 
      save_strategy= training_params.save_strategy, 
      save_steps= training_params.save_steps, 
      eval_steps= training_params.eval_steps, 
      greater_is_better= training_params.greater_is_better, 
      metric_for_best_model= training_params.metric_for_best_model, 
      save_total_limit= training_params.save_total_limit,  
      load_best_model_at_end= training_params.load_best_model_at_end  
      ##lr_scheduler_type="cosine",
      ##warmup_ratio = 0.1,

      # logging strategies 
      # remove_unused_columns=Falsegitp 
  )

  trainer = SFTTrainer(
      model=model,
      train_dataset=train_data,
      eval_dataset=val_data,
      dataset_text_field=training_params.dataset_text_field,
      peft_config=lora_config,
      args=training_arguments,
      max_seq_length = training_params.max_seq_length,
      # Currently (01/'24) Packing is not supported with Instruction Masking (data_collator argument is not supported with packing=True)
      # just packing without instruction masking gives good results already
      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), # see here: https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy#(optional)-preprocessing:-masking-instructions-by-using-the-datacollatorforcompletiononlylm
  #   packing=False,# True would create a ConstantLengthDataset so it can iterate over the dataset on fixed-length sequences
  #   neftune_noise_alpha=5,
  )

  from utils.wandb_callback import WandbPredictionProgressCallback
  progress_callback = WandbPredictionProgressCallback(
      trainer=trainer,
      tokenizer=tokenizer,
      val_dataset=test_data,
      num_samples=1,
      freq=1,
  )

  # Add the callback to the trainer
  trainer.add_callback(progress_callback)
  # with torch.autocast("cuda"):
  #   trainer.train()

  trainer.train()

  trainer.model.save_pretrained(f"{config.BASE_MODEL_CHECKPOINT.split('/')[1]}_prova") # save locally
  trainer.model.push_to_hub(ADAPTERS_CHECKPOINT, token=HF_TOKEN)

  wandb.finish()
  del model
  del trainer
  del tokenizer
  gc.collect()
  torch.cuda.empty_cache()

load_in_4bit_list = model_loading_params.load_in_4bit

bnb_4bit_quant_type_list = model_loading_params.bnb_4bit_quant_type
bnb_4bit_compute_dtype_list = model_loading_params.bnb_4bit_compute_dtype

llm_int8_threshold_list = model_loading_params.llm_int8_threshold

r_list = lora_params.r
lora_alpha_list = lora_params.lora_alpha
lora_dropout_list = lora_params.lora_dropout

gradient_accumulation_steps_list = training_params.gradient_accumulation_steps
learning_rate_list = training_params.learning_rate

for model_loading_params_idx in range(len(load_in_4bit_list)):
  
  load_in_4bit = load_in_4bit_list[model_loading_params_idx]
  load_in_8bit = not load_in_4bit
  bnb_4bit_quant_type = bnb_4bit_quant_type_list[model_loading_params_idx]
  bnb_4bit_compute_dtype = bnb_4bit_compute_dtype_list[model_loading_params_idx]
  llm_int8_threshold = llm_int8_threshold_list[model_loading_params_idx]
  print('I AM LOADING A MODEL IN load_in_4bit=', load_in_4bit, 'load_in_8bit=', load_in_8bit, 'bnb_4bit_quant_type=', bnb_4bit_quant_type, 'bnb_4bit_compute_dtype=', bnb_4bit_compute_dtype, 'llm_int8_threshold=', llm_int8_threshold)
  for r in r_list:
    for lora_alpha in lora_alpha_list:
      for lora_dropout in lora_dropout_list:
        for gradient_accumulation_steps in gradient_accumulation_steps_list:
          for learning_rate in learning_rate_list:
            nbits = 4
            if load_in_8bit:
              nbits = 8
            if not model_loading_params.quantization:
              nbits = "NoQuant"
            extra_str = ""
            if preprocessing_params.simplest_prompt:
              extra_str = "simplest_prompt_"
            else:
              extra_str = ""
            extra_str_cl = ""
            if preprocessing_params.clent:
              extra_str_cl += "_clent"
            ADAPTERS_CHECKPOINT = f"ferrazzipietro/{config.model_name}_{extra_str}adapters_{config.TRAIN_LAYER}_{nbits}_{bnb_4bit_compute_dtype}_{r}_{lora_alpha}_{lora_dropout}_{gradient_accumulation_steps}_{learning_rate}{extra_str_cl}"
            main(ADAPTERS_CHECKPOINT,
                  load_in_4bit, bnb_4bit_quant_type, bnb_4bit_compute_dtype, llm_int8_threshold,
                  r, lora_alpha, lora_dropout,
                  gradient_accumulation_steps,learning_rate)
            gc.collect()
            torch.cuda.empty_cache()
