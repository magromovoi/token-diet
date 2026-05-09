# Token Diet: Input and Output Compression for LLM API Cost Reduction

This project tests whether mechanical text compression can reduce LLM API costs on summarization without losing quality.

The experiment compares four conditions:

1. Baseline - raw reviews in, normal summary out.
2. Input compressed - strip filler words with spaCy before sending to the model.
3. Output compressed - prompt the model to respond in terse "caveman-speak".
4. Both - input stripping + caveman prompt combined.

## How to run

```bash
git clone git@github.com:magromovoi/token-diet.git
cd token-diet

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

export ANTHROPIC_API_KEY=your_key
python run.py
python visualize.py
```

## Hardware

Runs on any machine with internet access. No GPU needed - all computation is done via the Anthropic API. spaCy preprocessing runs on CPU.

## Dataset

[Amazon Reviews 2023 (on HuggingFace)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
