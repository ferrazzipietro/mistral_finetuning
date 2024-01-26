BASE_MODEL_CHECKPOINT="mistralai/Mistral-7B-Instruct-v0.2"
model_name=BASE_MODEL_CHECKPOINT.split('/')[1]
ADAPTERS_CHECKPOINT=f"ferrazzipietro/{model_name}_adapters" #"ferrazzipietro/mistral-7B-adapters-E3C-en-layer1"
FT_MODEL_CHECKPOINT="ferrazzipietro/ft_tmp" # "ferrazzipietro/mistral-7B-FT-E3C-en-layer1"

DATASET_CHEKPOINT="ferrazzipietro/e3c_finetuning_processed" #"ferrazzipietro/e3c_finetuning"
TRAIN_LAYER = "en.layer1"
train_on_subset = False
if train_on_subset:
    train_subset_size = 15