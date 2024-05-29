from datetime import datetime
from .preprocessing_params import simplest_prompt


tagging_label = 'TAG_3EPOCHS' # uso questo escamotage per passare il parametro a model_loading_params.py e quindi al nome del modello, considerato che non uso la quantizazione e quindi alcuni parametri posso sovrasciriverli


DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
BASE_MODEL_CHECKPOINT="meta-llama/Llama-2-7b-chat-hf" # "meta-llama/Llama-2-7b-chat-hf"  # 
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]

TRAIN_LAYER = "en.layer1"
ADAPTERS_CHECKPOINT= f"ferrazzipietro/{model_name}_adapters_{tagging_label}"
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" 
if simplest_prompt:
    ADAPTERS_CHECKPOINT=ADAPTERS_CHECKPOINT + "_simplest_prompt"

WANDB_PROJECT_NAME = f'finetune {model_name} {tagging_label}'
WANDB_RUN_NAME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

