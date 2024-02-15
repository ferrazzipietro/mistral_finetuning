from datetime import datetime

DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
BASE_MODEL_CHECKPOINT="mistralai/Mistral-7B-Instruct-v0.2"
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]

TRAIN_LAYER = "en.layer1"
ADAPTERS_CHECKPOINT= f"ferrazzipietro/{model_name}_adapters_{TRAIN_LAYER}"
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" 

WANDB_PROJECT_NAME = f'finetune {model_name} {TRAIN_LAYER}'
WANDB_RUN_NAME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
