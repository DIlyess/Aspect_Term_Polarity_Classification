from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import os

Label2Id = {"positive": 0, "negative": 1, "neutral": 2}
Id2Label = {0: "positive", 1: "negative", 2: "neutral"}

class Classifier:
    """
    The Classifier: complete the definition of this class template by completing the __init__() function and
    the 2 methods train() and predict() below. Please do not change the signature of these methods
     """


    ############################################# complete the classifier class below
    
    def __init__(self, ollama_url: str):
        self.model_name = "microsoft/deberta-v3-large"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # Ajouter les tokens spéciaux
        special_tokens = {'additional_special_tokens': ['[TERM]', '[/TERM]']}
        self.tokenizer.add_special_tokens(special_tokens)

        # checkpoint_path = "./results/checkpoint-752"

        # if os.path.exists(checkpoint_path):
        #     print(f"Chargement du modèle sauvegardé depuis {checkpoint_path}")
        #     self.model = AutoModelForSequenceClassification.from_pretrained(
        #         checkpoint_path,
        #         num_labels=3,
        #         label2id=Label2Id,
        #         id2label=Id2Label
        #     )
        # else:
        #     print("Aucun modèle sauvegardé trouvé — entraînement depuis zéro.")
        #     self.model = AutoModelForSequenceClassification.from_pretrained(
        #     self.model_name,
        #     num_labels=3,
        #     label2id=Label2Id,
        #     id2label=Id2Label
        #       )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3,
            label2id=Label2Id,
            id2label=Id2Label
            )

        # Redimensionner les embeddings pour les nouveaux tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

    def load_data(self, file_path):
        df = pd.read_csv(file_path, sep="\t", header=None)
        df.columns = ["label", "aspect", "term", "offset", "sentence"]
        return df
    
    def highlight_term(self, sentence, start, end):
        highlighted_sentence = sentence[:start] + "[TERM]" + sentence[start:end] + "[/TERM]" + sentence[end:]
        return highlighted_sentence
    
    def process_df(self, df):
        sentences = []
        aspects = []
        for _, row in df.iterrows():
            sentence = row["sentence"]
            aspect = row["aspect"]
            start, end = map(int, row["offset"].split(":"))
            highlighted_sentence = self.highlight_term(sentence, start, end)
            sentences.append(highlighted_sentence)
            aspects.append(aspect)
        tokenized_inputs = self.tokenizer(
            sentences,
            aspects,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        return tokenized_inputs
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, you must
          not train the model, and this method should contain only the "pass" instruction
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS

        """
        train_df = self.load_data(train_filename)
        dev_df = self.load_data(dev_filename)

        class PolarityDataset(Dataset):
            def __init__(self, dataframe, tokenizer, highlighter):
                self.dataframe = dataframe
                self.tokenizer = tokenizer
                self.highlighter = highlighter

            def __len__(self):
                return len(self.dataframe)
            
            def __getitem__(self, idx):
                row = self.dataframe.iloc[idx]
                start, end = map(int, row["offset"].split(":"))
                sentence = self.highlighter(row["sentence"], start, end)
                aspect = f"{row['aspect']} [SEP] {row['term']}"
                label = Label2Id[row["label"]]
                encoded = self.tokenizer(
                    sentence,
                    aspect,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                item = {k: v.squeeze() for k, v in encoded.items()}
                item["labels"] = torch.tensor(label)
                return item
            
        train_dataset = PolarityDataset(train_df, self.tokenizer, self.highlight_term)
        dev_dataset = PolarityDataset(dev_df, self.tokenizer, self.highlight_term)

        self.model.to(device)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_dir="./logs",
            logging_strategy="epoch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )

        trainer.train()


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, ignore the 'device'
        parameter (because the device is specified when launching the Ollama server, and not by the client side)
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        self.model.to(device)
        self.model.eval()

        df = self.load_data(data_filename)
        tokenized = self._prepare_batch(df)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            outputs = self.model(**tokenized)
            preds = torch.argmax(outputs.logits, dim=-1)

        return [Id2Label[i.item()] for i in preds]






