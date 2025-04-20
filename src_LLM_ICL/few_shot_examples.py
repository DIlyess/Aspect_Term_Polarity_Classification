import pandas as pd

train_filename = "../data/traindata.csv"
df_train = pd.read_csv(train_filename, header=None, sep="\t")
df_train.columns = ["sentiment", "aspect", "item", "position", "sentence"]


def sequential_prompt(item, position, aspect, sentence):
    return f"""You are a Aspect-Term Polarity Classifier for Sentiment Analysis.
            You have to predict the sentiment of a sentence given the aspect of an item and the position of the item in the sentence.
            The sentiment can be one of the following: "positive","neutral", "negative".
            Item: "{item}"
            Item Position: "{position}"
            Aspect: "{aspect}"
            Sentence: "{sentence}"
            Return just one word: "positive", "negative" or "neutral".
            """


def prompt_instruct(item, position, aspect, sentence):
    return f"""What is the sentiment of the word "{item}" at position "{position}" (regarding the criteria "{aspect}") in the sentence: "{sentence}"?
            Please return just one word: "positive", "negative" or "neutral".
            OUTPUT :
            """


EXAMPLES_INSTRUCT = [
    prompt_instruct(
        *df_train[["item", "position", "aspect", "sentence"]].iloc[0].values
    )
    + f"\nOUTPUT : {df_train['sentiment'].iloc[0]}"
]

EXAMPLES_SEQUENTIAL = [
    sequential_prompt(
        *df_train[["item", "position", "aspect", "sentence"]].iloc[0].values
    )
    + f"\nOUTPUT : {df_train['sentiment'].iloc[0]}"
]
