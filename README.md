# 🇮🇳 Low-Resource Indian Language Translation Using Transformers

A deep learning project focused on improving **English-to-Marathi machine translation** under limited training data conditions using a Transformer model implemented from scratch in PyTorch.

The project investigates how training data size, multilingual transfer learning, and back-translation affect translation quality for low-resource Indian languages.

## 📌 Project Overview

Machine translation between English and Indian languages remains challenging because of limited parallel training data, differences in sentence structure, and rich linguistic morphology.

This project explores these challenges by building a Transformer-based neural machine translation model and comparing its performance with pretrained translation models.

Rather than focusing only on translation accuracy, the project studies how models behave when training data is limited and analyzes common translation errors.

## 🎯 Objectives

- Implement the Transformer architecture from scratch using PyTorch.
- Develop an English-to-Marathi translation model.
- Measure performance with different training dataset sizes.
- Compare the custom model against pretrained translation models.
- Investigate back-translation and multilingual training.
- Analyze word-order differences between English and Marathi.
- Evaluate translations using BLEU and chrF++ scores.
- Build an interactive translation demo using Gradio.

## 🧠 Key Features

### 1. Transformer from Scratch
Implement the core Transformer components, including:

- Encoder and decoder layers
- Multi-head self-attention
- Positional encoding
- Feed-forward neural networks
- Residual connections and layer normalization
- Masking and autoregressive decoding

### 2. Data-Efficiency Experiments

Train and evaluate the model using different amounts of parallel training data.

| Training data | Purpose |
|---|---|
| 5,000 sentence pairs | Study extremely limited-data performance |
| 20,000 sentence pairs | Measure early learning improvements |
| 100,000 sentence pairs | Evaluate medium-scale training |
| 500,000 sentence pairs | Investigate performance at a larger scale |

The actual dataset sizes will depend on data availability and computational resources.

### 3. Low-Resource Improvement Techniques

**Back-translation:** Use monolingual Marathi text to generate synthetic training pairs and investigate whether additional data improves translation quality.

**Multilingual training:** Train on English-Hindi and English-Marathi data with language tags to study whether related-language information benefits Marathi translation.

**Pretrained baseline comparison:** Compare the custom Transformer with an appropriate pretrained model, such as IndicTrans2 or MarianMT, where supported.

### 4. Linguistic Error Analysis

Analyze common translation errors, including:

- Subject-verb-object (SVO) to subject-object-verb (SOV) reordering
- Gender agreement
- Postposition usage
- Honorifics and formality
- Rare words and vocabulary limitations

Attention visualizations will be used to investigate how the model aligns source words with translated words.

### 5. Translation Evaluation

Evaluate translation quality using:

- **BLEU:** Measures n-gram overlap between generated and reference translations.
- **chrF++:** Evaluates character-level and word-level overlap, useful for examining morphological variations.
- **FLORES-200:** A multilingual evaluation benchmark for comparable translation testing, subject to the selected language direction and split availability.

Results will be reported using actual experimental measurements rather than assumed improvements.

### 6. Interactive Demo

Build a Gradio web interface where users can:

1. Enter an English sentence.
2. Generate a Marathi translation.
3. Compare outputs from available models.
4. Explore examples of translation errors.

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **NLP:** Hugging Face Transformers, SentencePiece
- **Evaluation:** SacreBLEU, chrF++
- **Visualization:** Matplotlib, Seaborn (optional)
- **Demo:** Gradio
- **Development:** Jupyter Notebook / Google Colab / VS Code
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
low-resource-indian-translation/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── models/
│   ├── transformer.py
│   ├── encoder.py
│   ├── decoder.py
│   └── attention.py
│
├── tokenizer/
│   └── train_tokenizer.py
│
├── training/
│   ├── train.py
│   ├── back_translation.py
│   └── multilingual_training.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── error_analysis.py
│   └── visualize_attention.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

*This is a proposed structure; adjust the folders to match your actual implementation.*

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Samruddhi4849/low-resource-indian-translation.git
cd low-resource-indian-translation
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

Create a `requirements.txt` file with the libraries required by your implementation, for example:

```text
torch
numpy
pandas
matplotlib
sentencepiece
transformers
datasets
sacrebleu
gradio
```

Install them using:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your CUDA environment, if available.

## 🚀 Usage

The exact commands will depend on the scripts implemented in the repository.

### Train the Transformer

```bash
python training/train.py
```

### Evaluate translation quality

```bash
python evaluation/evaluate.py
```

### Launch the Gradio demo

```bash
python app.py
```

These commands are intended as a starting point and should be updated to match the actual filenames and command-line arguments in your project.

## 📊 Experimental Plan

The project will investigate the following research questions:

| Experiment | Research question |
|---|---|
| Data efficiency | How does translation quality change as training data increases? |
| Pretrained baseline | How does a from-scratch model compare with pretrained models? |
| Back-translation | Does synthetic parallel data improve Marathi translation? |
| Multilingual transfer | Can Hindi training data help English-to-Marathi translation? |
| Linguistic analysis | Which types of grammatical and word-order errors occur most often? |

### Results

Results will be added after experiments are completed.

| Model / experiment | Training data | BLEU | chrF++ |
|---|---:|---:|---:|
| Transformer from scratch | 5K pairs | TBD | TBD |
| Transformer from scratch | 20K pairs | TBD | TBD |
| Transformer from scratch | 100K pairs | TBD | TBD |
| Transformer from scratch | 500K pairs | TBD | TBD |
| Transformer + back-translation | TBD | TBD | TBD |
| Multilingual Transformer | TBD | TBD | TBD |
| Pretrained baseline | Model-dependent | TBD | TBD |

*All TBD values must be replaced with measured results. Ensure every model is evaluated on a consistent test set before comparing scores.*

## 🔬 Linguistic Analysis

English commonly follows SVO word order, while Marathi commonly follows SOV word order.

For example:

- **English:** I eat an apple.
- **Marathi:** मी सफरचंद खातो.

A translation model must learn to reorder words while preserving meaning and grammatical agreement.

The project will examine attention visualizations and manually inspect approximately 100 translated sentences to identify recurring errors and opportunities for improvement.

## 🌱 Future Improvements

- Compare BPE and SentencePiece unigram tokenization.
- Investigate romanized Marathi and code-mixed input.
- Test translation across government, news, and conversational domains.
- Explore COMET evaluation where suitable models and resources are available.
- Improve inference speed and deploy the demo publicly.

## 📈 Expected Outcomes

- A working Transformer implementation built from scratch.
- An experimental comparison across different data sizes.
- An evaluation of back-translation and multilingual training.
- Quantitative translation metrics and qualitative error analysis.
- An interactive English-to-Marathi translation demo.

The expected outcomes are research goals, not guaranteed performance improvements.

## 👩‍💻 Author

**Samruddhi Shinde**

Third-Year Information Technology Student

**Interests:** Artificial Intelligence, Natural Language Processing, Deep Learning, and Machine Translation.

## 📄 License

This project may be released under the MIT License. Add a `LICENSE` file before claiming that the repository is MIT-licensed.

## 🙌 Acknowledgements

- PyTorch
- Hugging Face
- FLORES-200 multilingual evaluation benchmark
- Research and open-source contributions to neural machine translation for Indian languages
