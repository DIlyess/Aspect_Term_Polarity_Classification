How to use LLM with ollama

# Setup

1. Install ollama from their [website](https://ollama.com/download)
2. Open a terminal and install the models

```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
```

3. Run the ollama server

```bash
ollama serve
```

if there is a port conflict, you can close ollama from your desktop, open a new terminal, and run the command above.

4. You can now run tester.py to test the model

```bash
python tester.py
```
