def input_shape(text: str) -> str:
    """
    Format the text to obtain the desired prompt.

    Args:
        text (str): The input text.

    Returns:
        str: A formatted string.
    """
    return f"<s>[INST] Extract the entities contained in this text: <<<{text}>>> [/INST]"

def output_shape_one_entity(offsets: str, text: str) -> str:
    """
    This function takes in two parameters, `offsets` and `text`, and returns a string that combines them in a specific format.
    
    Parameters:
        offsets (str): The offset value.
        text (str): The text value.
    
    Returns:
        str: A string that combines the offset and text values.
    """
    return f"offset: {offsets} text: {text}"

def output_shape_combine_entities(all_entities: list, separator: str) -> str:
    separator = ' ' + separator + ' ' 
    return separator.join([output_shape_one_entity(entity['offsets'], entity['text']) for entity in all_entities])