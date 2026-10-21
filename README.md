# 🇮🇳 Low-Resource English-to-Hindi Translation Using a Transformer From Scratch

**A PyTorch implementation of the Transformer architecture from “Attention Is All You Need,” investigating data efficiency in low-resource English-to-Hindi machine translation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)](https://pytorch.org/)
[![NLP](https://img.shields.io/badge/Domain-NLP-purple)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Status](https://img.shields.io/badge/Status-Experiments%20In%20Progress-orange)](.)

## 📌 Overview

Machine translation between English and Indian languages remains challenging because of limited parallel training data, differences in grammatical structure, and language-specific vocabulary.

This project implements the Transformer architecture from scratch in PyTorch and studies how the amount of training data affects English-to-Hindi translation quality.

Inspired by the paper *Attention Is All You Need*, the project uses a compact Transformer model trained on the IITB English-Hindi parallel corpus. Its performance is evaluated on FLORES-200 and compared with a pretrained translation baseline.

The main research question is:

> **How does the amount of parallel training data affect the translation quality of a Transformer trained from scratch, and how does it compare with a pretrained model?**

The project emphasizes reproducible experiments, quantitative evaluation, and honest analysis of model limitations rather than attempting to reproduce the original paper's benchmark scores.

## 🎯 Objectives

- Implement the core Transformer architecture from scratch using PyTorch.
- Study data efficiency across six training-set sizes, from 5,000 to 200,000 sentence pairs.
- Compare the custom Transformer with a pretrained translation model.
- Evaluate translation quality using BLEU, chrF++, and spBLEU.
- Investigate English SVO to Hindi SOV word-order reordering.
- Analyze attention patterns and common translation errors.
- Publish reproducible training scripts, evaluation results, and an interactive Gradio demo.

## 🧠 Model Architecture

The model follows the encoder-decoder Transformer architecture described in *Attention Is All You Need*.

### Implemented components

- Scaled dot-product attention
- Multi-head attention
- Padding masks and causal masks
- Sinusoidal positional encoding
- Position-wise feed-forward networks
- Encoder and decoder stacks
- Residual connections and layer normalization
- Shared source/target token embeddings and output projection weights
- Embedding scaling by √d_model
- Autoregressive decoding with greedy and beam search

Standard PyTorch building blocks such as `nn.Linear`, `nn.Embedding`, and `nn.LayerNorm` are used. The Transformer architecture and its attention computations are implemented by the project rather than delegated to a pretrained Transformer model.

### Configuration

| Parameter | Original Transformer Base | This Project |
|---|---:|---:|
| Encoder layers | 6 | 3 |
| Decoder layers | 6 | 3 |
| Model dimension | 512 | 256 |
| Attention heads | 8 | 4 |
| Feed-forward dimension | 2048 | 1024 |
| Dropout | 0.1 | 0.3 |
| Learning-rate warmup | 4,000 steps | 800 steps |
| Shared vocabulary | Approximately 37K | 8K |
| Tokenizer | BPE | SentencePiece BPE |

The compact configuration is designed to make experiments practical on a free Kaggle T4 GPU. It is a scaled-down implementation of the architecture, not an exact reproduction of the original model's capacity or training compute.

## 📚 Dataset and Preprocessing

### Training dataset

**IITB English-Hindi Parallel Corpus**

The corpus contains approximately 1.66 million English-Hindi sentence pairs. The project uses cleaned subsets of this corpus for training.

### Data preprocessing

The preprocessing pipeline applies the following steps:

1. Remove empty sentence pairs.
2. Filter sentences longer than 80 words.
3. Remove sentence pairs with length ratios outside 0.4–2.5.
4. Filter Hindi sentences that do not meet the project's Devanagari-script requirement.
5. Remove duplicate pairs.
6. Shuffle the cleaned dataset and create nested training subsets.

Each training subset is a prefix of the same shuffled dataset, allowing the effect of training-set size to be studied while keeping the data selection process consistent.

### Tokenization

A shared SentencePiece BPE tokenizer is trained for English and Hindi.

- Vocabulary size: 8,000 tokens
- Shared vocabulary for both languages
- Reserved token IDs for padding, unknown, beginning-of-sequence, and end-of-sequence tokens

**Important:** Every model checkpoint must be paired with the exact SentencePiece model used during its training. A tokenizer-checkpoint mismatch can produce severely degraded translations.

## ⚙️ Training Methodology

The training pipeline incorporates the following techniques:

- Adam optimizer with β₁ = 0.9, β₂ = 0.98, and ε = 1e-9
- Transformer learning-rate warmup schedule
- Label smoothing of 0.1
- Token-budget batching
- Dropout of 0.3
- Early stopping after five epochs without validation-loss improvement
- Validation-based checkpoint selection

Training runs are designed for a free Kaggle T4 GPU, with logs and checkpoints saved for each dataset size.

The pilot experiment uses approximately 18.7K cleaned training pairs. The full data-efficiency experiment is in progress.

## 📊 Data-Efficiency Experiment

The central experiment measures translation quality as the training dataset grows.

| Experiment | Training sentence pairs | Status |
|---|---:|---|
| Run 1 | 5,000 | In progress |
| Run 2 | 10,000 | In progress |
| Run 3 | 20,000 | In progress |
| Run 4 | 50,000 | In progress |
| Run 5 | 100,000 | In progress |
| Run 6 | 200,000 | In progress |

*Individual runs should be marked complete only after their final checkpoints and evaluation results have been saved.*

### Experimental controls

The following are held constant across runs:

- Model architecture and hyperparameters
- Tokenizer and vocabulary
- Learning-rate schedule
- Validation set of 500 sentence pairs
- Data ordering, using nested prefixes of one shuffled file

The primary variable is the number of training sentence pairs.

The resulting data-efficiency curve will help show how quickly the model learns from additional data and whether improvements diminish as the dataset grows.

## 📏 Evaluation Metrics

The model is evaluated using the FLORES-200 English-Hindi benchmark.

| Metric | Purpose |
|---|---|
| BLEU | Measures n-gram overlap between generated and reference translations |
| chrF++ | Measures character-level and word-level overlap |
| spBLEU | Uses multilingual tokenization for more comparable evaluation |
| Validation loss | Tracks prediction error on held-out training-domain data |
| Perplexity | Expresses the model's predictive uncertainty through exponentiated loss |

### Evaluation protocol

- **FLORES dev:** Used for development, validation of decoding choices, and intermediate evaluation.
- **FLORES devtest:** Reserved for final reporting and evaluated once after development decisions are complete.
- **Pilot evaluation:** The initial scores were calculated on only the first 100 sentences of the dev split and are not directly comparable to full-split results.

The final report will distinguish pilot measurements from full-split evaluation results.

## 📈 Current Pilot Results

The initial pilot run used approximately 18.7K cleaned training pairs.

| Metric | Pilot result |
|---|---:|
| Best validation loss | Approximately 4.56 |
| Perplexity | Approximately 96 |
| BLEU, first 100 dev sentences | 0.46 |
| chrF++, same 100 sentences | 10.36 |
| spBLEU, same 100 sentences | 0.49 |
| Output/reference length ratio | 0.938 |
| Beam search, beam size 4 | Not yet measured |

**Interpretation:** The pilot model has learned some target-language patterns, but its translation quality remains very weak. Its outputs can be fluent Hindi without accurately preserving the English input's meaning.

Validation loss reached its minimum near epoch 36 while training loss continued to decline, suggesting overfitting after that point.

These measurements are preliminary. The 100-sentence sample is too small to establish reliable benchmark performance, and the scores must not be presented as final FLORES-200 results.

## 🔬 Planned Research Extensions

### 1. Pretrained baseline comparison

Compare the from-scratch model against a suitable pretrained English-to-Hindi translation model, such as MarianMT or IndicTrans2, where a compatible model and evaluation setup are available.

Both models should be evaluated on the same FLORES sentences with consistent reference translations and metric settings.

The comparison will quantify the performance gap and help explain the advantages of pretrained linguistic knowledge.

### 2. Low-resource training technique

Investigate at least one approach for improving translation with limited parallel data:

- **Back-translation:** Generate synthetic English-Hindi training pairs using monolingual Hindi text.
- **Multilingual training:** Investigate whether training with another Indian language can improve Hindi translation.

Any improvement will be reported using measured scores. Negative results will also be documented.

### 3. Attention visualization and error analysis

Inspect attention heatmaps to investigate how the decoder aligns English source words with Hindi output tokens.

Manually analyze approximately 100 translated sentences and categorize errors such as:

- Subject-object-verb reordering
- Incorrect grammatical agreement
- Missing or incorrect postpositions
- Incorrect handling of names and rare words
- Hallucinated content and loss of source meaning

Attention weights are diagnostic visualizations, not definitive proof that the model has learned a particular linguistic rule.

### 4. Gradio demo

Develop a simple interactive interface that allows users to:

1. Enter an English sentence.
2. Generate a Hindi translation.
3. View the output from the trained model.
4. Compare results with a pretrained baseline, if integrated.
5. Inspect selected examples of translation errors.

## 🧪 Testing

The implementation currently has **23 passing tests**, covering implemented model and training components.

Testing focuses on verifying:

- Attention output shapes and masking behavior
- Encoder and decoder tensor dimensions
- Positional encoding
- Padding and causal masking
- Forward-pass behavior
- Training and loss calculations
- Decoding behavior
- Tokenizer and checkpoint compatibility

Tests validate implementation behavior; passing tests alone do not establish translation quality.

## 📂 Repository Structure

The following is a suggested logical structure. Update the paths to match the actual repository files.

```text
low-resource-en-hi-transformer/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── tokenizer/
│   ├── train_tokenizer.py
│   └── spm.model
├── models/
│   ├── attention.py
│   ├── positional_encoding.py
│   ├── encoder.py
│   ├── decoder.py
│   └── transformer.py
├── training/
│   ├── train.py
│   └── scheduler.py
├── evaluation/
│   ├── evaluate.py
│   ├── error_analysis.py
│   └── visualize_attention.py
├── tests/
│   └── ...
├── experiments/
│   ├── results.csv
│   └── plots/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

Large datasets and model checkpoints should not be committed directly to Git unless the repository is specifically configured to host them. Document how to obtain required artifacts.

## 🚀 Setup and Usage

### Prerequisites

- Python
- PyTorch
- SentencePiece
- SacreBLEU
- NumPy
- A GPU runtime for practical training at larger dataset sizes

### Installation

Clone the repository:

```bash
git clone https://github.com/Samruddhi4849/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Training and evaluation

Use the actual script names and arguments from the repository. A typical workflow is:

```bash
python training/train.py
python evaluation/evaluate.py
python app.py
```

These commands are illustrative and should be adjusted to match the implemented entry points. Training requires the preprocessed dataset and the matching tokenizer artifacts.

For full experiments, follow the repository's Kaggle notebook instructions and use the saved checkpoints and logs for reproducibility.

## ⚠️ Risks and Limitations

- **Limited compute:** The compact model is trained with significantly less compute than the original Transformer Base.
- **Small-data overfitting:** A model trained on a limited number of sentence pairs may memorize patterns without learning reliable translation.
- **Domain mismatch:** IITB training text and FLORES evaluation text differ in domain and style.
- **Tokenizer limitations:** The shared tokenizer was trained using a limited sample and may be suboptimal for larger training subsets.
- **Runtime interruptions:** Kaggle session limits can interrupt training; checkpoints and logs should be saved regularly.
- **Slow decoding:** CPU inference can be slow, especially for beam search.
- **Experimental variability:** A single run per dataset size may produce a noisy data-efficiency curve.
- **Evaluation leakage:** The devtest split must not be used to tune hyperparameters or select the best checkpoint.

The objective is to understand the behavior and limitations of a small from-scratch model, not to claim production-level translation performance.

## 🗺️ Project Roadmap

- [x] Implement the core Transformer architecture
- [x] Implement the training pipeline
- [x] Implement greedy and beam-search decoding
- [x] Add BLEU and chrF++ evaluation
- [x] Pass 23 implementation tests
- [x] Complete the initial pilot run
- [ ] Complete all six data-efficiency runs
- [ ] Evaluate a pretrained baseline on the same test sentences
- [ ] Investigate one low-resource training technique
- [ ] Complete attention visualization and manual error analysis
- [ ] Finalize FLORES devtest results
- [ ] Build the Gradio demo
- [ ] Publish final plots, tables, and reproduction instructions

## 📚 References

1. Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
2. [IIT Bombay English-Hindi Parallel Corpus](https://www.cfilt.iitb.ac.in/iitb_parallel/).
3. [FLORES-200: A Multilingual Evaluation Benchmark](https://github.com/facebookresearch/flores).
4. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html).
5. [SentencePiece](https://github.com/google/sentencepiece).
6. [SacreBLEU](https://github.com/mjpost/sacrebleu).

## 👩‍💻 Author

**Samruddhi Shinde**

B.Tech — Information Technology

Project area: Natural Language Processing, Neural Machine Translation, Deep Learning, and Transformer Architectures.

## 📄 License

This project's source code is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Samruddhi Shinde

The MIT License applies to the project's code and does not automatically apply to third-party datasets, pretrained models, or other external resources.

### Dataset Licenses

- **IITB English-Hindi Parallel Corpus:** Subject to the dataset's original terms of use, including any non-commercial research restrictions.
- **FLORES-200:** Subject to its applicable license and usage conditions. Please refer to the [official FLORES repository](https://github.com/facebookresearch/flores) for details.

Users are responsible for complying with the applicable licenses and usage restrictions of all third-party resources.


---

*This project is an educational and experimental implementation. Final benchmark results, pretrained-model comparisons, and claimed improvements will be added only after the corresponding experiments are completed.*
