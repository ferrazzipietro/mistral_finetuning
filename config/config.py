BASE_MODEL_CHECKPOINT="mistralai/Mistral-7B-v0.1"
FT_MODEL_CHECKPOINT="ferrazzipietro/mistral-7B-E3C-FT"

DATASET_CHEKPOINT="ferrazzipietro/e3c_finetuning_processed" #"ferrazzipietro/e3c_finetuning"
TRAIN_LAYER = "it.layer1"
train_on_subset = True
if train_on_subset:
    train_subset_size = 15