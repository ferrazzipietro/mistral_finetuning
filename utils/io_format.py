
def input_shape(text: str) -> str:
    return f"<s>[INST] Extract the entities contained in this text: <<<{text}>>> [/INST]"

def output_shape_one_entity(offsets: str, text: str) -> str:
    return f"offset: {offsets} text: {text}"

def output_shape_combine_entities(all_entities: list, separator: str) -> str:
    separator = ' ' + separator + ' ' 
    return separator.join([output_shape_one_entity(entity['offsets'], entity['text']) for entity in all_entities])
