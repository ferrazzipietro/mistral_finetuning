from datetime import datetime


DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
BASE_MODEL_CHECKPOINT="meta-llama/Llama-2-7b-chat-hf" # "meta-llama/Llama-2-7b-chat-hf"  # 
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]

TRAIN_LAYER = "en.layer1"
ADAPTERS_CHECKPOINT= f"ferrazzipietro/{model_name}_adapters_{TRAIN_LAYER}"
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" 

WANDB_PROJECT_NAME = f'finetune {model_name} {TRAIN_LAYER}'
WANDB_RUN_NAME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


slovenian_train_path = 'mistral_finetuning/data/slovenian/E3C_Slovenian_Train_SL_L1.csv'
slovenian_test_path = 'mistral_finetuning/data/slovenian/E3C_Slovenian_Test_SL_L1.csv'

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
r = [16, 32, 64] # [16, 32, 64] reduce the number to finish faster
lora_alpha = [16,32] #[32, 64] 
lora_dropout = [0.02] # [0.05, 0.01]
bias =  "lora_only" 
use_rslora = True
task_type="CAUSAL_LM"
target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]# substituted by the function find_all_linear_names()
import torch

"""
    bnb_4bit_quant_type (str, optional, defaults to "nf4") – The quantization type to use. Options are "nf4" and "fp4".
    bnb_4bit_compute_dtype (torch.dtype, optional, defaults to torch.bfloat16) – This sets the computational type which might be different than the input tipe
    bnb_4bit_use_double_quant (bool, optional, defaults to True) – Whether to use double quantization.
"""
"""
    llm_int8_threshold (float, optional, defaults to 6.0) – This corresponds to the outlier threshold for outlier detection as described. 
                                                            Any hidden states value that is above this threshold will be considered an outlier 
                                                            and the operation on those values will be done in fp16defaults to 6.0.
    llm_int8_skip_modules (List[str], optional, defaults to ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]) – An explicit list of the modules 
                                                                                                                    that we do not want to convert in 8-bit
    llm_int8_has_fp16_weight (bool, optional, defaults to False) – This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning 
                                                                    as the weights do not have to be converted back and forth for the backward pass.
"""
torch_dtype=torch.float16
quantization = False
load_in_4bit=[False]
bnb_4bit_quant_type = ["nf4"]
bnb_4bit_compute_dtype = [torch.bfloat16]
llm_int8_threshold = [6.0]

bnb_4bit_use_double_quant = True
llm_int8_has_fp16_weight = True
llm_int8_skip_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]



offset=False
instruction_on_response_format='Extract the CLINICAL ENTITIES contained in the text. Do not extract any entity which is not clinical.\nReturn the result in a json format: [{"entity":"clinical_entity_name"}].'
simplest_prompt=False
clent = True

### TrainingArguments
### TrainingArguments
num_train_epochs= 1
per_device_train_batch_size= 16
gradient_accumulation_steps= [1]#[2,4,8] # reduce the number to finish faster
optim = "paged_adamw_8bit"
learning_rate= [2e-4]
weight_decay= 0.001
fp16= False 
bf16= True
max_grad_norm= 0.3
max_steps= -1
warmup_ratio= 0.3
group_by_length= True
lr_scheduler_type= "constant"

logging_steps=2
logging_strategy="steps"
evaluation_strategy= "steps"
save_strategy=evaluation_strategy
save_steps= 5
eval_steps=save_steps
greater_is_better=False
metric_for_best_model="eval_loss"
save_total_limit = 1
load_best_model_at_end = True

### SFTTrainer
"""
    max_seq_length - The maximum sequence length to use for the ConstantLengthDataset and for automatically creating the Dataset. Defaults to 512.
    dataset_text_field - The name of the field containing the text to be used for the dataset. Defaults to "text".
    packing - Used only in case dataset_text_field is passed. This argument is used by the ConstantLengthDataset to pack the sequences of the dataset.
"""
max_seq_length= 1024
dataset_text_field="prompt"
packing=False
