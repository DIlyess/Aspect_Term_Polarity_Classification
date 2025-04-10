from typing import List
from ollama import Client
import torch


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
        #ollama_url= "http://your_ollama_server_url:11434"
        model_name='llama3.2'
        self.client = Client(base_url=ollama_url)
        self.model_name = model_name


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
        pass

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
        predictions = []
   
        with open(data_filename, 'r') as file:
            for line in file:
                sentence, aspect_term, aspect_category = line.strip().split('\t')[4], line.strip().split('\t')[2], line.strip().split('\t')[1]
                prompt = f"Phrase : {sentence}\nTerme : {aspect_term}\nCatégorie : {aspect_category}\nPolarité :"
                response = self.client.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}])
                predictions.append(response['message']['content'].strip())

        return predictions