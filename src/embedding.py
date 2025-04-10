import torch
from torch.utils.data import Dataset
import pandas as pd

class Embedding(Dataset):
    def __init__(self, filename, tokenizer):
        self.data = pd.read_csv(filename, sep='\t',header=None) 
        self.data.columns=['polarity','aspect_category','target_term','character_offsets','sentence']
        self.tokenizer = tokenizer
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extraction des informations de la ligne correspondante
        sentence = self.data.iloc[idx]['sentence']
        aspect_term = self.data.iloc[idx]['target_term']
        aspect_category = self.data.iloc[idx]['aspect_category']
        polarity = self.data.iloc[idx]['polarity']
        aspect_term_offsets = self.data.iloc[idx]['character_offsets']

        # Conversion de la polarité en étiquette numérique
        label = self.label_map[polarity]

        # Extraction des offsets de caractères
        start_offset, end_offset = map(int, aspect_term_offsets.split(':'))

        # Encodage des entrées pour le modèle
        # Ici, nous incluons la phrase, le terme d'aspect, la catégorie d'aspect et les offsets
        # Vous pouvez adapter la stratégie d'encodage en fonction de votre modèle et de vos besoins
        encoding = self.tokenizer(
            sentence,
            aspect_term + " " + aspect_category + f" {start_offset}:{end_offset}",
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        # Retourne les entrées encodées et l'étiquette
        return {key: val.squeeze(0) for key, val in encoding.items()}, torch.tensor(label)
