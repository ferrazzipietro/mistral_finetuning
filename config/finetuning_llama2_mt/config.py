from datetime import datetime
from .preprocessing_params import simplest_prompt


DATASET_CHEKPOINT="ferrazzipietro/mt_samples_problems" 
BASE_MODEL_CHECKPOINT="meta-llama/Llama-2-7b-chat-hf" # "meta-llama/Llama-2-7b-chat-hf"  # 
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]

TRAIN_LAYER = "MT"
ADAPTERS_CHECKPOINT= f"ferrazzipietro/{model_name}_adapters_{TRAIN_LAYER}"
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" 
if simplest_prompt:
    ADAPTERS_CHECKPOINT=ADAPTERS_CHECKPOINT + "_simplest_prompt"

WANDB_PROJECT_NAME = f'finetune {model_name} {TRAIN_LAYER}'
WANDB_RUN_NAME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

