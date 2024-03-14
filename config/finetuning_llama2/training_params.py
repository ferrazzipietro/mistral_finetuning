### TrainingArguments
num_train_epochs= 3
per_device_train_batch_size= 2
gradient_accumulation_steps= [2,4]#[2,4,8]
optim = "paged_adamw_8bit"
save_steps= 1000
logging_strategy="steps"
logging_steps= 10
learning_rate= [2e-4]
weight_decay= 0.001
fp16= True 
bf16= False
max_grad_norm= 0.3
max_steps= -1
warmup_ratio= 0.3
group_by_length= True
lr_scheduler_type= "constant"


### SFTTrainer
"""
    max_seq_length - The maximum sequence length to use for the ConstantLengthDataset and for automatically creating the Dataset. Defaults to 512.
    dataset_text_field - The name of the field containing the text to be used for the dataset. Defaults to "text".
    packing - Used only in case dataset_text_field is passed. This argument is used by the ConstantLengthDataset to pack the sequences of the dataset.
"""
max_seq_length= 1024
dataset_text_field="prompt"
packing=False
