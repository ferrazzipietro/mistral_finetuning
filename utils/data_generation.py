from pandas import DataFrame
from utils.io_format import input_shape, output_shape_combine_entities 
import deprecation

class DataGenerator():
    
    def __init__(self, input_max_length: int=8096) -> None:
        self.input_max_length = input_max_length

    def assess_max_length(self):
        """
        Assess the maximum length of the input and output of the dataset

        Returns:

        """
        pass

    @deprecation.deprecated(details="Use the prompt function instead")
    def concatenate_text_entities(self, data, split: str, enitites_separator_in_output: str)-> DataFrame:
        """
        Format 

        Args:
            split: the split name
            enitites_separator_in_output: the separator to be used in the output

        Returns:
            the formatted dataset
        """
        lang, layer = split.split('.', 1)  
        if layer in ['layer1', 'layer2']:
            input = [input_shape(text) for text in data['text']]
            output = [output_shape_combine_entities(entities, separator=enitites_separator_in_output)  + ' </s>' for entities in data['entities']]
            processed_df = DataFrame({'input': input, 'output': output, 'language': lang, 'layer': layer})
        else:
            input = data['text']
            output = [output_shape_combine_entities(entities, separator=enitites_separator_in_output) for entities in data['entities']]
            processed_df = DataFrame({'input': input, 'output': output, 'language': lang, 'layer': layer})
        return processed_df
    
    def split_into_sentences(self, data, split: str, resolve_multiple_annotations: bool=True) -> DataFrame:
        """
        Split each dataset's row into as many rows as sentences in the text

        Args:
            data: a split of the dataset
            split: the split name
            resolve_multiple_annotations: in case the same word is annotated multiple times, return it only once
        Returns:
            the dataset with the text split into sentences
        """
        _, layer = split.split('.', 1)  

        if layer in ['layer1', 'layer2', 'layer2.validation']:
            sentences = []
            entities_in_sentences = []
            original_text = []
            original_id = []
            for k in range(len(data['text'])):
                entities = data['entities'][k]
                for passage in data['passages'][k]:
                    sentences.append(passage['text'])
                    original_text.append(data['text'][k])
                    original_id.append(data['original_id'][k])
                    sentence_span = passage['offsets']
                    entities_in_passage = []
                    for entity in entities:
                        if sentence_span[0] <= entity['offsets'][0] and entity['offsets'][1] <= sentence_span[1]:
                            already_seen = False
                            if resolve_multiple_annotations:
                                for entity_in_sentence in entities_in_passage:
                                    if entity_in_sentence['offsets'] == entity['offsets'] and entity_in_sentence['text'] == entity['text']:
                                        already_seen = True
                            if not already_seen:
                                entities_in_passage.append(entity)
                    entities_in_sentences.append(entities_in_passage)
            out = DataFrame({'sentence': sentences, 'entities': entities_in_sentences, 'original_text':original_text, 'original_id':original_id})  

        if layer == 'layer3':
            empty_entity_list = self._entity_structure()
            sentences = ['' for _ in range(data.num_rows)]
            entities_in_sentences = [empty_entity_list for _ in range(data.num_rows)]
            original_text = [text for text in data['text']]
            original_id = [id for id in data['original_id']]
            out =  DataFrame( {'sentence': sentences, 
                                       'entities': entities_in_sentences, 
                                        'original_text': original_text, 
                                        'original_id': original_id})
        return out
    
    def _entity_structure(self):
        """
        Return the structure of an empty entity for layer 3, required to be consistent with the format of layers 1 and 2
        """
        entity_structure = [{'id': '',
                    'offsets': [-1,-1],
                    'role': '',
                    'semantic_type_id': '',
                    'text': '',
                    'type': ''}]
        return entity_structure