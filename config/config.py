DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" 
TRAIN_LAYER = "it.layer1"

ADAPTERS_VERSION = "_v0.2"
BASE_MODEL_CHECKPOINT="mistralai/Mistral-7B-Instruct-v0.2"
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]
ADAPTERS_CHECKPOINT=f"ferrazzipietro/{model_name}_adapters_{TRAIN_LAYER}_{ADAPTERS_VERSION}" 
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" 