import wandb
from tqdm import tqdm
import torch
from transformers import GenerationConfig
from transformers import TrainerCallback
from transformers.integrations import WandbCallback



class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each 
    logging step during training. It allows to visualize the 
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset 
          for generating predictions.
        num_samples (int, optional): Number of samples to select from 
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=100, freq=2):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated 
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from 
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
        if state.epoch % self.freq == 0:
            # generate predictions
            predictions = self.trainer.predict(self.sample_dataset)
            # decode predictions and labels
            predictions = decode_predictions(self.tokenizer, predictions)
            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"sample_predictions": records_table})



class PrinterCallback(TrainerCallback):

    def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        super().__init__()
        self._log_model = log_model
        self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
    
    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
    
    def samples_table(self, examples):
        """
        Create a wandb.Table to store the generations
        """
        records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            prompt = example["text"]
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
        return records_table
        
    def on_evaluate(self, args, state, control,  **kwargs):
        """
        Log the wandb.Table after calling trainer.evaluate
        """
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})



# class LLMSampleCB(WandbCallback):
#     def __init__(self, trainer, test_dataset, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
#         super().__init__()
#         self._log_model = log_model
#         self.sample_dataset = test_dataset.select(range(num_samples))
#         self.model, self.tokenizer = trainer.model, trainer.tokenizer
#         self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
#                                                            max_new_tokens=max_new_tokens)
#     def generate(self, prompt):
#         tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
#         with torch.inference_mode():
#             output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
#         return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
    
#     def samples_table(self, examples):
#         """
#         Create a wandb.Table to store the generations
#         """
#         records_table = wandb.Table(columns=["prompt", "generation"] + list(self.gen_config.to_dict().keys()))
#         for example in tqdm(examples, leave=False):
#             prompt = example["text"]
#             generation = self.generate(prompt=prompt)
#             records_table.add_data(prompt, generation, *list(self.gen_config.to_dict().values()))
#         return records_table
        
#     def on_evaluate(self, args, state, control,  **kwargs):
#         """
#         Log the wandb.Table after calling trainer.evaluate
#         """
#         super().on_evaluate(args, state, control, **kwargs)
#         records_table = self.samples_table(self.sample_dataset)
#         self._wandb.log({"sample_predictions":records_table})
