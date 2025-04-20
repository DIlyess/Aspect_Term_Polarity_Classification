# Aspect-Based Sentiment Classifier with Transformers and LLMs

Contributors : Maria Roman, Hana Naji, Ilyess Doragh

This repository implements a sentiment classifier for **Aspect-Based Sentiment Analysis (ABSA)**. It includes **two approaches**:

1. **Fine-tuning a Transformer-based classifier** using the DeBERTa-v3-large model.
2. **In-context learning (ICL) using LLMs** via the [Ollama](https://ollama.com) framework and models such as `gemma3:1b` and `gemma3:4b`.

## Task Description

The classifier takes as input:

- A **sentence**
- An **aspect category**
- A **term** within the sentence referring to the aspect (highlighted by an offset)

and outputs a **sentiment label** (`positive`, `negative`, `neutral`) expressing the polarity of the aspect in the context of the sentence.

Example input:

```json
{
  "sentence": "The menu looked great, and the waiter was very nice, but when the food came, it was average.",
  "aspect": "FOOD#STYLE_OPTIONS",
  "term": "menu",
  "position": "4:8",
  "sentiment": "positive"
}
```

## Implemented Approaches

### 1. Fine-Tuned Transformer Classifier

This method trains a DeBERTa-v3-large model from HuggingFace on a custom ABSA dataset.

- **Model**: `microsoft/deberta-v3-large`
- **Feature Representation**:
  - Input : `beginning of sentence... [TERM]term[/TERM] end of sentence...` and `aspect`
  - These two elements were passed to an automatic tokenizer from pretrained DeBERTa-v3-large.
  - Special tokens `[TERM]` and `[/TERM]` were added to highlight the relevant phrase in the sentence.
- **Training**:
  - Fine-tuning with `Trainer` from HuggingFace Transformers.
  - Used `4` epochs, batch size `8`, and learning rate `1e-5`.
  - The classifier is evaluated on a dev set with ~90% accuracy.
- **Resources**:
  - Trained on GPU (CUDA) if available.
  - Libraries: `transformers`, `torch`, `datasets`, `sentencepiece`, `lightning`.

**Accuracy on Dev Set**: **90.00%**

### 2. In-Context Learning with Ollama LLMs

This method avoids training and instead uses few-shot prompting to query a local LLM through the [Ollama](https://ollama.com) server.

- **Models Used**:
  - `gemma3:1b`
  - `gemma3:4b`
- **Prompting Strategy**:
  - Includes few-shot examples with either:
    - A **single-instruction style prompt**
    - A **step-by-step sequential reasoning prompt**
  - Models are instructed to return structured sentiment output in JSON format.
- **No training required**, predictions are made through prompts.

**Accuracy on Dev Set**: **65.00%**

---

## Setup

#### Install Dependencies

```bash
pip install -r requirements.txt
```

### For Transformer approach

You just need to run the `tester.py` script, which will load the model and make predictions.

### For LLM approach

#### 1. Install Ollama

Download and install from [Ollama's official website](https://ollama.com/download)

#### 2. Download LLMs

```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
```

#### 3. Start the Ollama server

```bash
ollama serve
```

#### 4.Run the tester

```bash
python tester.py
```
