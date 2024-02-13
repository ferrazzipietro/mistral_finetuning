Use python 3.8

install this extra package:
pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda113

# STRUCTURE
This repo contains the code to 
1) LORA Fine-Tune an LLM using the transformer library (Hugging Face)
2) Do inference on the e3c dataset 
3) Postprocess the output models at inference time, in order to extract the entities in the right format
2) Evaluate the performances of the FineTuned model