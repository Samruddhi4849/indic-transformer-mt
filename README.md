# 🇮🇳 Low-Resource English-to-Hindi Translation Using a Transformer From Scratch

**A PyTorch implementation of the Transformer architecture from “Attention Is All You Need,” investigating data efficiency in low-resource English-to-Hindi machine translation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)](https://pytorch.org/)
[![NLP](https://img.shields.io/badge/Domain-NLP-purple)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Tests](https://img.shields.io/badge/Tests-23%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experiments%20In%20Progress-orange)](.)

## 📌 Overview

Machine translation between English and Indian languages remains challenging because of limited parallel training data, differences in grammatical structure, and language-specific vocabulary.

This project implements the Transformer architecture from scratch in PyTorch and studies how the amount of training data affects English-to-Hindi translation quality.

Inspired by the paper *Attention Is All You Need*, the project uses a compact Transformer model trained on the IITB English-Hindi parallel corpus. Its performance is evaluated on FLORES-200.

The main research question is:

> **How does the amount of parallel training data affect the translation quality of a Transformer trained from scratch?**

The project emphasizes reproducible experiments, quantitative evaluation, and honest analysis of model limitations rather than attempting to reproduce the original paper's benchmark scores.

## 🎯 Objectives

- Implement the core Transformer architecture from scratch using PyTorch.
- Study data efficiency across training-set sizes from 5,000 to 200,000 sentence pairs (5K to 50K completed so far).
- Compare the custom Transformer with a pretrained translation model *(planned)*.
- Evaluate translation quality using BLEU, chrF++, and spBLEU.
- Investigate English SVO to Hindi SOV word-order reordering *(planned)*.
- Analyze attention patterns and common translation errors *(planned)*.
- Publish reproducible training scripts, evaluation results, and an interactive demo *(planned)*.

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
- Shared embedding and output projection weights
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
| Parameters | about 65M | 7,577,600 |

The compact configuration is designed to make experiments practical on a free Kaggle T4 GPU. It is a scaled-down implementation of the architecture, not an exact reproduction of the original model's capacity or training compute.

## 📚 Dataset and Preprocessing

### Training dataset

**IITB English-Hindi Parallel Corpus**: approximately 1.66 million English-Hindi sentence pairs. The project uses cleaned subsets of this corpus.

### Data preprocessing

1. Remove empty sentence pairs.
2. Filter sentences longer than 80 words.
3. Remove sentence pairs with length ratios outside 0.4–2.5.
4. Drop pairs whose Hindi side contains no Devanagari characters.
5. Remove duplicate pairs.
6. Shuffle the cleaned dataset (fixed seed) and create nested training subsets.

Each training subset is a prefix of the same shuffled dataset, so the 50K set contains the 20K set, and so on. The effect of training-set size is then studied with a consistent data selection process.

On the pilot's 20K sample, cleaning kept 18,820 pairs (94%). Bad length ratios were the main reason for removal.

### Tokenization

A shared SentencePiece BPE tokenizer is trained for English and Hindi.

- Vocabulary size: 8,000 tokens
- Shared vocabulary for both languages
- Reserved token IDs for padding, unknown, beginning-of-sequence, and end-of-sequence tokens

**Important:** Every model checkpoint must be paired with the exact SentencePiece model used during its training. A mismatch silently produces garbage (this happened once during the pilot: validation loss jumped from 4.56 to 7.82).

## ⚙️ Training Methodology

- Adam optimizer with β₁ = 0.9, β₂ = 0.98, and ε = 1e-9
- Transformer learning-rate warmup schedule
- Label smoothing of 0.1
- Token-budget batching (sentences of similar length grouped together)
- Dropout of 0.3
- Early stopping after a patience window (configurable) and validation-based checkpoint selection

Training runs on a free Kaggle T4 GPU, with logs and checkpoints saved for each dataset size. Training on a CPU is impractical: one epoch on 5K pairs took about 20 minutes on a laptop.

## 📊 Data-Efficiency Experiment

The central experiment measures translation quality as the training dataset grows.

| Run | Training pairs | Status |
|---|---:|---|
| 1 | 5,000 | ✅ Completed |
| 2 | 10,000 | ✅ Completed |
| 3 | 20,000 | ✅ Completed |
| 4 | 50,000 | ✅ Completed |
| 5 | 100,000 | Not run |
| 6 | 200,000 | Not run |

### Experimental controls

Held constant across runs: model architecture and hyperparameters, tokenizer and vocabulary, learning-rate schedule, a validation set of 500 sentence pairs, and the data ordering (nested prefixes of one shuffled file). The primary variable is the number of training sentence pairs.

### Results

Each model was trained for up to 30 epochs and scored with greedy decoding on the **first 200 sentences of the FLORES-200 `dev` split**.

| Training pairs | Optimizer steps | Best val loss | Best val perplexity | chrF++ | BLEU | spBLEU | Output/ref length |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 1,200 | 5.405 | 222.5 | 7.07 | 0.27 | 0.23 | 1.15 |
| 10,000 | 2,370 | 4.918 | 136.7 | 7.39 | 0.14 | 0.13 | 1.97 |
| 20,000 | 4,680 | 4.468 | 87.2 | 11.99 | 0.30 | 0.31 | 1.34 |
| 50,000 | 11,760 | 3.878 | 48.3 | 12.97 | 0.89 | 0.79 | 1.17 |

![Data-efficiency curve](docs/data_efficiency.png)

*Left: validation perplexity against training-set size (log scale). Right: FLORES dev scores (first 200 sentences, greedy decoding).*

### What the results show

- **Validation perplexity falls steadily with more data**: from 222.5 at 5K pairs to 48.3 at 50K, about 4.6 times lower.
- **chrF++ rises from about 7 to 13**, with the 5K and 10K points close together.
- **BLEU stays below 1 at every size.** On 200 sentences at this score level it moves mostly by noise, so no trend should be read into it.
- **Even at 50K pairs, translations are weak.** The model produces fluent-looking Hindi that often repeats itself or drifts away from the source sentence. The 10K model's outputs are almost twice the reference length (ratio 1.97) because of repetition loops, which distorts its scores.
- A from-scratch Transformer needs far more data than this to translate reliably.

## 📈 Pilot Results

An earlier pilot run used about 18.7K cleaned training pairs with its own tokenizer, before the controlled experiment above.

| Metric | Pilot result |
|---|---:|
| Best validation loss | about 4.56 |
| Perplexity | about 96 |
| BLEU, first 100 dev sentences, greedy | 0.46 |
| chrF++, same 100 sentences | 10.36 |
| spBLEU, same 100 sentences | 0.49 |
| Output/reference length ratio | 0.938 |
| Beam search, beam size 4 | Not yet measured |

Validation loss reached its minimum near epoch 36 while training loss kept falling, which shows overfitting after that point.

![Pilot training curves](docs/loss_curves.png)

The pilot and the experiment use different tokenizers, data samples and evaluation sentences (100 versus 200), so their scores are **not directly comparable**. The scores are preliminary and must not be read as full-split FLORES-200 results.

## 📏 Evaluation Metrics

| Metric | Purpose |
|---|---|
| BLEU | Measures n-gram overlap between generated and reference translations |
| chrF++ | Measures character-level and word-level overlap; more forgiving for Hindi's rich morphology |
| spBLEU | BLEU with a multilingual tokenizer, for more comparable results |
| Validation loss | Tracks prediction error on held-out training-domain data |
| Perplexity | Exponentiated validation loss |

### Evaluation protocol

- **FLORES dev:** used for development and intermediate evaluation.
- **FLORES devtest:** reserved for the final report and evaluated once, after all development decisions are made. It has not been used yet.

## 🔬 Planned Extensions

1. **Pretrained baseline comparison:** score a pretrained model (such as MarianMT or IndicTrans2) on the same FLORES sentences with the same metric settings, and explain the gap.
2. **Low-resource technique:** investigate back-translation or multilingual training, and report measured gains, including negative results.
3. **Attention visualization and error analysis:** inspect attention heatmaps for English SVO to Hindi SOV reordering, and manually categorize errors (reordering, agreement, postpositions, rare words, hallucination). Attention weights are diagnostic visualizations and not proof of a learned linguistic rule.
4. **Larger runs:** the 100K and 200K training sizes.
5. **Faster decoding:** KV-caching, since decoding currently recomputes the full prefix at every step.
6. **Interactive demo** with Gradio.

## 🧪 Testing

The implementation has **23 passing tests**:

```bash
uv run pytest -v
```

They verify:

- Attention output shapes, row sums, and agreement with PyTorch's reference attention
- Causal masking (changing future tokens leaves earlier outputs unchanged) and padding masking
- Positional encoding values and the relative-offset property
- Exact parameter counts for the encoder and decoder layers
- Weight sharing between the embeddings and the output layer
- The learning-rate schedule peaking exactly at the warmup step
- Batching: every pair used once, token budget respected, correct padding
- Beam search with beam size 1 matching greedy decoding

Passing tests validate the implementation. They do not establish translation quality.

## 📂 Repository Structure

```text
indic-transformer-mt/
├── model/               # attention, layers, encoder, decoder, transformer, masks, decoding
├── training/            # dataset batching and learning-rate schedule
├── tests/               # unit tests
├── results/
│   ├── pilot_results.csv
│   └── exp/             # data-efficiency results (summary.csv, results.csv)
├── docs/                # figures
├── prepare_flores.py    # downloads FLORES-200
├── prepare_data.py      # pilot: downloads IITB, makes a 20K subset
├── clean_data.py        # pilot: filtering
├── train_tokenizer.py   # pilot: SentencePiece
├── make_subsets.py      # experiment: 200K cleaned pairs and tokenizer
├── train.py             # training
├── evaluate.py          # beam/greedy decoding, BLEU, chrF++
├── plot_data_efficiency.py
├── plot_curves.py
├── check_ckpt.py        # sanity check: checkpoint vs tokenizer
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

Datasets, tokenizers and checkpoints are not stored in the repository. The scripts download or rebuild them.

## 🚀 Setup and Usage

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A GPU runtime (such as a free Kaggle T4) for practical training

### Installation

```bash
git clone https://github.com/Samruddhi4849/indic-transformer-mt.git
cd indic-transformer-mt
uv sync
uv run pytest -v
```

### Reproduce the data-efficiency experiment

```bash
uv run python prepare_flores.py
uv run python make_subsets.py

# train and score one size (GPU recommended)
uv run python train.py --data data/train200k --spm data/spm_exp.model --size 50000 --tag n50000 --epochs 30 --patience 4
uv run python evaluate.py --spm data/spm_exp.model --ckpt checkpoints/n50000.pt --tag n50000 --beam 1 --n 200

# plot the curve from the saved results
uv run python plot_data_efficiency.py
```

Repeat the `train.py` and `evaluate.py` lines for sizes 5000, 10000 and 20000. The results used here were produced on Kaggle and are saved in `results/exp/`.

## ⚠️ Risks and Limitations

- **Limited compute:** a compact model, trained on a free GPU with much less compute than the original Transformer Base.
- **Data size and training steps changed together:** the 5K run got 1,200 optimizer steps and the 50K run got 11,760, so the curve mixes both effects.
- **No early stopping triggered:** every run ended at the 30-epoch cap (best epoch 28 to 30), so the larger runs were probably still improving.
- **Single run per size:** no repeated seeds, so the curve is noisy. Dropout was fixed at 0.3, which suits small data better than large.
- **Small evaluation sample:** 200 sentences with greedy decoding. BLEU is especially unreliable at these scores.
- **Domain mismatch:** IITB training text (government and web) differs from FLORES (Wikipedia-style).
- **Tokenizer limitations:** trained on only 20K pairs, which may limit larger runs.
- **Slow decoding:** CPU decoding takes about 3 seconds per sentence because there is no key-value caching.
- **Evaluation leakage:** the devtest split must not be used for tuning or checkpoint selection.

The objective is to understand the behavior and limits of a small from-scratch model, not to claim production-level translation.

## 🗺️ Project Roadmap

- [x] Implement the core Transformer architecture
- [x] Implement the training pipeline
- [x] Implement greedy and beam-search decoding
- [x] Add BLEU, chrF++ and spBLEU evaluation
- [x] Pass 23 implementation tests
- [x] Complete the pilot run
- [x] Run the data-efficiency experiment for 5K, 10K, 20K and 50K pairs
- [ ] Run the 100K and 200K sizes
- [ ] Evaluate a pretrained baseline on the same sentences
- [ ] Investigate one low-resource training technique
- [ ] Attention visualization and manual error analysis
- [ ] Final FLORES devtest results
- [ ] Build the Gradio demo
- [ ] Add key-value caching for faster decoding

## 📚 References

1. Vaswani, A. et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
2. Kunchukuttan, A. et al. (2018). [The IIT Bombay English-Hindi Parallel Corpus](https://www.cfilt.iitb.ac.in/iitb_parallel/). LREC 2018.
3. NLLB Team. [FLORES-200](https://github.com/facebookresearch/flores).
4. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html).
5. [SentencePiece](https://github.com/google/sentencepiece).
6. [SacreBLEU](https://github.com/mjpost/sacrebleu).

## 👩‍💻 Author

**Samruddhi Shinde**

B.Tech, Information Technology

Project area: Natural Language Processing, Neural Machine Translation, Deep Learning, and Transformer Architectures.

## 📄 License

This project's source code is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Samruddhi Shinde

The MIT License applies to the project's code and does not automatically apply to third-party datasets, pretrained models, or other external resources.

### Dataset Licenses

- **IITB English-Hindi Parallel Corpus:** subject to its original terms of use, including its non-commercial research restriction.
- **FLORES-200:** subject to its applicable license. Refer to the [official FLORES repository](https://github.com/facebookresearch/flores) for details.

Users are responsible for complying with the licenses and usage restrictions of all third-party resources.

---
