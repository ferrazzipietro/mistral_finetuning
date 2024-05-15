from datetime import datetime
from .preprocessing_params import simplest_prompt

DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
BASE_MODEL_CHECKPOINT = "sapienzanlp/Minerva-3B-base-v1.0" #  "mistralai/Mistral-7B-v0.1"  # "mistralai/Mistral-7B-v0.1" # "mistralai/Mistral-7B-Instruct-v0.2"
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]

TRAIN_LAYER = "it.layer1"
ADAPTERS_CHECKPOINT= f"ferrazzipietro/{model_name}_adapters_{TRAIN_LAYER}"

if simplest_prompt:
    ADAPTERS_CHECKPOINT=ADAPTERS_CHECKPOINT + "_simplest_prompt"
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" 

WANDB_PROJECT_NAME = f'finetune {model_name} {TRAIN_LAYER}'
WANDB_RUN_NAME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")