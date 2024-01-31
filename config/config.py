DATASET_CHEKPOINT="ferrazzipietro/e3c-sentences" #"ferrazzipietro/e3c_finetuning"
TRAIN_LAYER = "it.layer1"

BASE_MODEL_CHECKPOINT="mistralai/Mistral-7B-Instruct-v0.2"
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]
ADAPTERS_CHECKPOINT=f"ferrazzipietro/{model_name}_adapters_{TRAIN_LAYER}" #"ferrazzipietro/mistral-7B-adapters-E3C-en-layer1"
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" # "ferrazzipietro/mistral-7B-FT-E3C-en-layer1"
