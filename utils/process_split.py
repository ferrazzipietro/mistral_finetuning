from pandas import DataFrame
from .io_format import input_shape, output_shape_combine_entities

class Process_split():
    def __init__(self, data: DataFrame, input_max_length: int=8096) -> None:
        self.data = data
        self.input_max_length = input_max_length

    def assess_max_length(self):
        """
        Assess the maximum length of the input and output of the dataset

        Returns:

        """

        pass

    def apply(self, split: str, enitites_separator_in_output: str)-> DataFrame:
        """
        Apply the formatting function to the dataset

        Args:
            split: the split name
            enitites_separator_in_output: the separator to be used in the output

        Returns:
            the formatted dataset
        """
        lang, layer = split.split('.', 1)  
        if layer in ['layer1', 'layer2']:
            input = [input_shape(text) for text in self.data[split]['text']]
            output = [output_shape_combine_entities(entities, separator=enitites_separator_in_output)  + ' </s>' for entities in self.data[split]['entities']]
            processed_df = DataFrame({'input': input, 'output': output, 'language': lang, 'layer': layer})
        else:
            input = self.data[split]['text']
            output = [output_shape_combine_entities(entities, separator=enitites_separator_in_output) for entities in self.data[split]['entities']]
            processed_df = DataFrame({'input': input, 'output': output, 'language': lang, 'layer': layer})
        return processed_df
