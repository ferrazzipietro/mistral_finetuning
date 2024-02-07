from datetime import datetime
from lora_params import r

DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
TRAIN_LAYER = "es.layer1"
BASE_MODEL_CHECKPOINT="mistralai/Mistral-7B-Instruct-v0.2"
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]
ADAPTERS_CHECKPOINT=f"ferrazzipietro/{model_name}_adapters_{TRAIN_LAYER}_loraR{r}" 
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" 

WANDB_PROJECT_NAME = f'finetune {model_name} {TRAIN_LAYER}'
WANDB_RUN_NAME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class tmp():
    def __init__(self):
        self.BASE_MODEL_CHECKPOINT = BASE_MODEL_CHECKPOINT
        self.DATASET_CHEKPOINT = DATASET_CHEKPOINT
        self.TRAIN_LAYER = TRAIN_LAYER
        self.FT_MODEL_CHECKPOINT = FT_MODEL_CHECKPOINT
        self.WANDB_PROJECT_NAME = WANDB_PROJECT_NAME
        self.WANDB_RUN_NAME = WANDB_RUN_NAME
    
