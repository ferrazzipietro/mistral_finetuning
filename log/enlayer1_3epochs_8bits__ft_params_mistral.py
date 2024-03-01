from datetime import datetime

DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
BASE_MODEL_CHECKPOINT="mistralai/Mistral-7B-Instruct-v0.2"
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]

TRAIN_LAYER = "en.layer1"
ADAPTERS_CHECKPOINT= f"ferrazzipietro/{model_name}_adapters_{TRAIN_LAYER}"
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" 

WANDB_PROJECT_NAME = f'finetune {model_name} {TRAIN_LAYER}'
WANDB_RUN_NAME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

"""
    r – the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters
    lora_alpha – scaling factor, expressed in int. Higher alpha results in larger update matrices with more trainable parameters
    lora_dropout – dropout probability in the lora layers
    bias – the type of bias to use. Options are "none", "all", "lora_only". If 'all' or 'lora_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
    use_rslora - whether to use the rank-scaled version of LoRA (i.e., sets the adapter scaling factor to `lora_alpha/math.sqrt(r)` 
                instead of `lora_alpha/r`)
    target_modules - The names of the modules to apply the adapter to. If None, automatic.
"""
r = [16, 32, 64]
lora_alpha = [32]
lora_dropout = [0.05, 0.01]
bias =  "lora_only" 
use_rslora = True
task_type="CAUSAL_LM"
target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]# substituted by the function find_all_linear_names()

import torch


load_in_4bit=[False]
bnb_4bit_quant_type = ["nf4"]
bnb_4bit_compute_dtype = [torch.bfloat16]
llm_int8_threshold = [6.0]

bnb_4bit_use_double_quant = True
llm_int8_has_fp16_weight = True
llm_int8_skip_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]


from transformers import AutoTokenizer

task='finetuning'
offset=False
instruction_on_response_format='Extract the entities contained in the text and the offset, i.e. the position of that entity in the string. Extract only entities contained in the text.\nReturn the result in a json format.'
n_shots = 0
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CHECKPOINT)
list_of_examples=[]
list_of_responses=[]

### TrainingArguments
num_train_epochs= 3
per_device_train_batch_size= 8
gradient_accumulation_steps= [2,4,8]
optim = "paged_adamw_8bit"
save_steps= 1000
logging_strategy="steps"
logging_steps= 10
learning_rate= [2e-4, 8e-4]
weight_decay= 0.001
fp16= True 
bf16= False
max_grad_norm= 0.3
max_steps= -1
warmup_ratio= 0.3
group_by_length= True
lr_scheduler_type= "constant"


### SFTTrainer
"""
    max_seq_length - The maximum sequence length to use for the ConstantLengthDataset and for automatically creating the Dataset. Defaults to 512.
    dataset_text_field - The name of the field containing the text to be used for the dataset. Defaults to "text".
    packing - Used only in case dataset_text_field is passed. This argument is used by the ConstantLengthDataset to pack the sequences of the dataset.
"""
max_seq_length= 1024
dataset_text_field="prompt"
packing=False


