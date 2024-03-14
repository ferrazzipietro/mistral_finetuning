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
r = [8, 16]
lora_alpha = [16, 32]
lora_dropout = [0.01] # [0.05, 0.01]
bias =  "lora_only" 
use_rslora = True
task_type="CAUSAL_LM"
target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]# substituted by the function find_all_linear_names()
