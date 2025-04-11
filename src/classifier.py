from typing import List
import ollama
import pandas as pd
import torch

from few_shot_examples import EXAMPLES


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
        self.ollama_url = ollama_url  # Store Ollama server URL
        self.client = ollama.Client(ollama_url)
        self.examples = EXAMPLES  # Stores few-shot learning examples
        self.model = "gemma3:4b"  # Specify the model to use

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
        df = pd.read_csv(data_filename, sep="\t")

        predictions = []
        for i, row in df.iterrows():
            sentiment = row.iloc[0]
            aspect = row.iloc[1]
            item = row.iloc[2]
            position = row.iloc[3]
            sentence = row.iloc[4]

            # Construct the prompt
            prompt = f""" 
                        You are a Aspect-Term Polarity Classifier for Sentiment Analysis.
                        You have to predict the sentiment of a sentence given the aspect of an item and the position of the item in the sentence.
                        The sentiment can be one of the following: "positive","neutral", "negative".
                        Item: "{item}"
                        Item Position: "{position}"
                        Aspect: "{aspect}"
                        Sentence: "{sentence}"
                        Return just one word: "positive", "negative" or "neutral".
                      """

            # Query Ollama
            response = self.client.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            sentiment_prediction = response["message"]["content"].strip()
            predictions.append(sentiment_prediction)

        return predictions
