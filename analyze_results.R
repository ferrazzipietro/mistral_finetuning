# Load the ggplot2 library
library(ggplot2)
library(tidyverse)

data <- read.csv("/Users/pietroferrazzi/Desktop/dottorato/mistral_finetuning/data/evaluation_results/joint_results.csv")
data %>% head(3)

show_results_grouped <- function(data){
  cols <- c('maxNewTokensFactor', 'nShotsInference', 'nbit', 'r', 'lora_alpha', 'lora_dropout', 'gradient_accumulation_steps', 'learning_rate')
  for (i in 1:length(cols)){
    print(cols[i])
    res <- data %>% 
      #filter(model==model_name) %>%
      group_by(model,
               !!sym(cols[i])) %>%
      summarise(f1=mean(f1_score),
                precision=mean(precision),
                recall=mean(recall))
    print(res)
  }
}

show_results_grouped(data)

