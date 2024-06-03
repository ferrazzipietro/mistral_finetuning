"""
    r – the rank of the update matrices, expressed in int. Lower rank results in smaller update matrices with fewer trainable parameters
    lora_alpha – scaling factor, expressed in int. Higher alpha results in larger update matrices with more trainable parameters
    lora_dropout – dropout probability in the lora layers
    bias – the type of bias to use. Options are "none", "all", "lora_only". If 'all' or 'lora_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
    use_rslora - whether to use the rank-scaled version of LoRA (i.e., sets the adapter scaling factor to `lora_alpha/math.sqrt(r)` 
                instead of `lora_alpha/r`)
    target_modules - The names of the modules to apply the adapter to. If None, automatic.
"""
r = [16] # [16, 32, 64] reduce the number to finish faster
lora_alpha = [32] 
lora_dropout = [0.01] # [0.05, 0.01]
bias =  "lora_only" 
use_rslora = True
task_type="CAUSAL_LM"
target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]# substituted by the function find_all_linear_names()

import torch

"""
    bnb_4bit_quant_type (str, optional, defaults to "nf4") – The quantization type to use. Options are "nf4" and "fp4".
    bnb_4bit_compute_dtype (torch.dtype, optional, defaults to torch.bfloat16) – This sets the computational type which might be different than the input tipe
    bnb_4bit_use_double_quant (bool, optional, defaults to True) – Whether to use double quantization.
"""
"""
    llm_int8_threshold (float, optional, defaults to 6.0) – This corresponds to the outlier threshold for outlier detection as described. 
                                                            Any hidden states value that is above this threshold will be considered an outlier 
                                                            and the operation on those values will be done in fp16defaults to 6.0.
    llm_int8_skip_modules (List[str], optional, defaults to ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]) – An explicit list of the modules 
                                                                                                                    that we do not want to convert in 8-bit
    llm_int8_has_fp16_weight (bool, optional, defaults to False) – This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning 
                                                                    as the weights do not have to be converted back and forth for the backward pass.
"""
torch_dtype=torch.float16
quantization = False
load_in_4bit=[False]
bnb_4bit_quant_type = [tagging_label]
bnb_4bit_compute_dtype = [tagging_label]
llm_int8_threshold = [6.0]

bnb_4bit_use_double_quant = True
llm_int8_has_fp16_weight = True
llm_int8_skip_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]

