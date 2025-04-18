from typing import List
from embedding import *
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import copy

class Classifier:
    """
    The Classifier: complete the definition of this class template by completing the __init__() function and
    the 2 methods train() and predict() below. Please do not change the signature of these methods
    """

    ############################################# complete the classifier class below

    def __init__(self, ollama_url: str):
        """
        This should create and initilize the model.
        !!!!! If the approach you have choosen is in-context-learning with an LLM from Ollama, you should initialize
         the ollama client here using the 'ollama_url' that is provided (please do not use your own ollama
         URL!)
        !!!!! If you have choosen an approach based on training an MLM or a generative LM, then your model should
        be defined and initialized here.
        """
        model_name = 'bert-base-uncased'  # Utilisation d'un modèle BERT pré-entraîné autorisé
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

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

        train_dataset = Embedding(train_filename, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        train_dataset = Embedding(train_filename, self.tokenizer)
        val_data_loader = DataLoader(dev_filename, batch_size=16, shuffle=False)

        self.model.to(device)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        loss_fn = torch.nn.CrossEntropyLoss()


        epochs=10
        persistence=4 #max number of epochs we continue training without improving accuracy 
        max_acc, max_f1, max_epoch= 0, 0, 0
        best_model = copy.deepcopy(self.model)

        for epoch in range(epochs):  # Nombre d'époques
            for batch in train_loader:
                inputs, labels = batch
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()

             #Validation
            acc, f1 = self.evaluate_acc_f1(val_data_loader)
            print('> val_acc: {:.4f}, val_f1: {:.4f}'.format(acc, f1))
            
            #Save model with best validation accuracy
            if acc > max_acc:
                max_acc = acc
                max_epoch = epoch
                best_model = copy.deepcopy(self.model)

            if f1 > max_f1:
                max_f1 = f1

            if epoch - max_epoch >= persistence:
                print('>> early stop.')
                break
            
            print("Best Validation Accuracy : {:.4f}".format(max_acc))
            print("Best F1-score : {:.4f}".format(max_f1))
            self.model = best_model


        

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

        dataset =Embedding(data_filename, self.tokenizer)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

        predictions = []
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

        with torch.no_grad():
            for batch in data_loader:
                inputs, _ = batch
                inputs = {key: val.to(device) for key, val in inputs.items()}

                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend([label_map[pred.item()] for pred in preds])

        return predictions