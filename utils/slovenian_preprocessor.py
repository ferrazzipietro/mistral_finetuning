import pandas as pd
import string
from datasets import Dataset

class Slovenian_preprocessor():

    def __init__(self, data):
        self.data = data
    
    def preprocess(self):
        self.data['label'] = self.data['label'].astype(str)
        text = ''
        overall_entities = []
        entities = []
        current_entity = ''
        prev_word=''
        for _, row in self.data.iterrows():
            # print(f"word: {row['word']}, label: {row['label']}, prev_word: {prev_word}")
            if prev_word == '.' and len(text.split())>5:
                if current_entity:
                    entities.append({'entity': current_entity})
                    current_entity = ''
                # print(f"ENTRO NELLA COND")
                overall_entities.append({'text': text, 'entities': entities})
                text = ''
                entities = []
            word, label = row['word'], row['label']
            if text!='' or (word.strip() in string.punctuation) or (prev_word.strip() in string.punctuation):
                space = ' '
            else:
                space = ''
            text = text + space + word
            if label != 'O':
                if label.startswith('B-'):
                    if current_entity:
                        entities.append({'entity': current_entity})
                    current_entity = word
                elif label.startswith('I-') and current_entity:
                    current_entity += ' ' + word
            prev_word = word
        data_df = pd.DataFrame(columns=['text', 'entities'])
        for el in overall_entities:
            el['text'] = el['text'].strip() 
            data_df = pd.concat([data_df, pd.DataFrame([el])], ignore_index=True)
        self.data = Dataset.from_pandas(data_df, split='train')
