Use python 3.8

install this extra package:
pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda113

# STRUCTURE
This repo contains the code to 
1) LORA Fine-Tune an LLM using the transformer library (Hugging Face). Refer to `finetuning_v2.py`
2) Do inference on the e3c dataset and postprocess the responses. Refer to `postprocessing.ipynb`
3) Evaluate the performances of the FineTuned model. Refer to `evaluation.ipynb`
du -h --max-depth=1