import torch

quantization = False
load_in_4bit=[True, False]
bnb_4bit_quant_type = ["nf4","nf4"]
bnb_4bit_compute_dtype = [torch.bfloat16, torch.bfloat16]
llm_int8_threshold = [6.0, 6.0]

bnb_4bit_use_double_quant = True
llm_int8_has_fp16_weight = True
llm_int8_skip_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]





# load_in_4bit=True
# load_in_8bit=False

# bnb_4bit_quant_type, bnb_4bit_compute_dtype, bnb_4bit_use_double_quant = "", torch.bfloat16, True
# llm_int8_threshold, llm_int8_skip_modules, llm_int8_has_fp16_weight = 0.0, [], True

# if load_in_4bit:
#     """
#         bnb_4bit_quant_type (str, optional, defaults to "nf4") – The quantization type to use. Options are "nf4" and "fp4".
#         bnb_4bit_compute_dtype (torch.dtype, optional, defaults to torch.bfloat16) – This sets the computational type which might be different than the input tipe
#         bnb_4bit_use_double_quant (bool, optional, defaults to True) – Whether to use double quantization.
#     """
#     bnb_4bit_quant_type = "nf4"
#     bnb_4bit_compute_dtype = torch.bfloat16
#     bnb_4bit_use_double_quant = True

# if load_in_8bit:
#     """
#         llm_int8_threshold (float, optional, defaults to 6.0) – This corresponds to the outlier threshold for outlier detection as described. 
#                                                                 Any hidden states value that is above this threshold will be considered an outlier 
#                                                                 and the operation on those values will be done in fp16defaults to 6.0.
#         llm_int8_skip_modules (List[str], optional, defaults to ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]) – An explicit list of the modules 
#                                                                                                                         that we do not want to convert in 8-bit
#         llm_int8_has_fp16_weight (bool, optional, defaults to False) – This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning 
#                                                                         as the weights do not have to be converted back and forth for the backward pass.
#     """
#     llm_int8_threshold = 6.0
#     # # substituted by the function find_all_linear_names() # llm_int8_skip_modules = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
#     llm_int8_has_fp16_weight = True
