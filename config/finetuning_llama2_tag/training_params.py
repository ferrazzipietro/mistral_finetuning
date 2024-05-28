### TrainingArguments
num_train_epochs= 8
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

logging_steps=1
logging_strategy="steps"
evaluation_strategy= "steps"
save_strategy=evaluation_strategy
save_steps= 10
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
