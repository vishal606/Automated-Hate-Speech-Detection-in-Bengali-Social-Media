# 🇧🇩 BanglaGuard
### A Comprehensive NLP Pipeline for Bengali Hate Speech Detection
### Using Classical ML, Deep Learning & Transformer Models

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-yellow?style=for-the-badge&logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-1.23%2B-013243?style=for-the-badge&logo=numpy)
![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-green?style=for-the-badge)
![BERT](https://img.shields.io/badge/BERT-bangla--bert--base-red?style=for-the-badge)

</div>

---

## 📌 Project Overview

**BanglaGuard** is a complete end-to-end NLP research project that applies **26 natural language processing techniques** directly on real Bengali text to detect hate speech. Starting from raw Bengali Unicode sentences, the project walks through every major NLP concept — from basic text normalization all the way to Transformer self-attention — and benchmarks 8 different ML/DL classification models.

> **Task:** Binary Classification → Hate Speech (1) vs Non-Hate Speech (0)  
> **Language:** Bengali (বাংলা) — 8th most spoken language in the world  
> **Dataset:** 30,000 real Bengali sentences collected from social media

---

## 📂 Repository Structure

```
BanglaGuard/
│
├── 📓 Bengali_NLP_Complete.ipynb   ← Main notebook (26 sections, 70 cells)
├── 📊 Bengali_hate_speech_.csv     ← Dataset (30,000 Bengali sentences)
└── 📄 README.md                    ← This file
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Total Samples** | 30,000 sentences |
| **Language** | Bengali (বাংলা) |
| **Classes** | Hate (1) = 10,000 · Non-Hate (0) = 20,000 |
| **Class Ratio** | 33.3% Hate · 66.7% Non-Hate |
| **Categories** | 7 topics |
| **Avg Words/Sentence** | ~14.7 |
| **Source** | Bengali social media (Facebook, YouTube comments) |

### Category Distribution

| Category | Samples | Hate % |
|----------|---------|--------|
| Crime | 4,763 | ~38% |
| Entertainment | 4,527 | ~31% |
| Sports | 4,406 | ~35% |
| Religion | 4,370 | ~42% |
| Meme, TikTok & Others | 4,236 | ~29% |
| Politics | 3,987 | ~36% |
| Celebrity | 3,711 | ~27% |

---

## 🗂️ Notebook Sections — 26 NLP Techniques

### 🔍 Text Preprocessing

| # | Section | Description |
|---|---------|-------------|
| 1 | **EDA** | Class balance, category heatmaps, word frequency, length distributions |
| 2 | **Text Normalization** | Remove URLs, @mentions, keep Bengali Unicode (U+0980–U+09FF) |
| 3 | **Tokenization & Stopwords** | Bengali whitespace tokenizer + custom 60-word stopword list |
| 4 | **Stemming** | Rule-based suffix stripper for Bengali verbal & nominal inflections |
| 5 | **Lemmatization** | Dictionary + POS-aware suffix rules → morphological root forms |
| 6 | **POS Tagging** | Heuristic tagger: N / V / ADJ / ADV / PRON / CONJ / PART |
| 7 | **TBT (Brill Tagger)** | 6 context-based correction rules applied iteratively |

### 📐 Feature Representations

| # | Section | Description |
|---|---------|-------------|
| 8 | **One-Hot Encoding** | Binary word vectors + multi-hot document representation |
| 9 | **Bag of Words (BoW)** | Count matrix, 5,000 Bengali features, discriminative word analysis |
| 10 | **N-Gram Features** | Uni/bi/trigrams per class + chi² ranking of hate phrases |
| 11 | **TF-IDF** | Word-level (1–2 gram) + char-level (2–4 gram) TF-IDF matrices |
| 12 | **LSA** | TruncatedSVD, 2D semantic space, category clustering |
| 13 | **Word2Vec** | Skip-gram trained on 30K Bengali sentences, PCA visualization |
| 14 | **GloVe** | Bengali co-occurrence matrix + SVD embeddings |
| 15 | **BERT** | `sagorsarker/bangla-bert-base` fine-tuning complete code |

### 🤖 Traditional Machine Learning

| # | Model | Features | Notes |
|---|-------|----------|-------|
| 16 | **Naive Bayes** | Char TF-IDF | Complement NB for class imbalance |
| 17 | **Logistic Regression** | Char TF-IDF | ROC + Precision-Recall curves |
| 18 | **SVM (LinearSVC)** | Char TF-IDF | Decision boundary distribution |
| 19 | **Random Forest** | Char TF-IDF | Feature importances, 200 trees |

### 🧠 Deep Learning (NumPy from Scratch)

| # | Model | Architecture | Notes |
|---|-------|-------------|-------|
| 20 | **ANN** | Input(256) → Dense(128,ReLU) → Dense(64,ReLU) → Sigmoid | Xavier init, mini-batch SGD |
| 21 | **RNN** | Input(16) → Elman RNN(64) → Sigmoid | Hidden state propagation |
| 22 | **LSTM** | Input(16) → LSTM(64) → Sigmoid | All 4 gates: Forget/Input/Cell/Output |
| 23 | **GRU** | Input(16) → GRU(64) → Sigmoid | 2 gates: Update + Reset |
| 24 | **Seq2Seq** | LSTM Encoder → Bahdanau Attention → Decoder | Attention weight heatmaps |
| 25 | **Transformer** | 4-Head Self-Attention on Word2Vec | Attention per Bengali token |

### 🏆 Final Comparison

| # | Section | Description |
|---|---------|-------------|
| 26 | **Grand Comparison** | Accuracy bar chart, radar chart, ranked table, 8 confusion matrices |

---

## 📈 Results

| Rank | Model | Accuracy | F1-Hate | Type |
|------|-------|----------|---------|------|
| 🥇 1 | BERT (bangla-bert-base) | ~93.2% | ~0.908 | Transformer |
| 🥈 2 | SVM (LinearSVC) | ~89.4% | ~0.830 | Traditional ML |
| 🥉 3 | Logistic Regression | ~88.7% | ~0.821 | Traditional ML |
| 4 | Naive Bayes | ~85.1% | ~0.790 | Traditional ML |
| 5 | Random Forest | ~84.6% | ~0.773 | Traditional ML |
| 6 | ANN (NumPy) | ~83.5% | ~0.751 | Deep Learning |
| 7 | LSTM (NumPy) | ~81.8% | ~0.734 | Deep Learning |
| 8 | GRU (NumPy) | ~81.2% | ~0.729 | Deep Learning |
| 9 | RNN (NumPy) | ~79.4% | ~0.698 | Deep Learning |

> 💡 **Key Finding:** Character-level TF-IDF (2–4 gram) + SVM gives the best accuracy without GPU.  
> BERT achieves ~4% improvement but requires GPU for practical training time.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BanglaGuard.git
cd BanglaGuard

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn gensim nltk

# Optional: for BERT fine-tuning
pip install transformers torch accelerate datasets
```

---

## 🚀 Quick Start

```python
# 1. Open the notebook
jupyter notebook Bengali_NLP_Complete.ipynb

# 2. Place dataset in the same folder
#    Bengali_hate_speech_.csv

# 3. Run All Cells (Kernel → Restart & Run All)
```

### Run on Google Colab
```python
# Upload both files to Colab, then:
!pip install gensim scikit-learn pandas numpy matplotlib seaborn nltk -q

import pandas as pd
df = pd.read_csv("Bengali_hate_speech_.csv")
print(f"Loaded {len(df):,} Bengali sentences")
```

---

## 📦 Dependencies

```
pandas >= 1.5.0
numpy >= 1.23.0
matplotlib >= 3.6.0
seaborn >= 0.12.0
scikit-learn >= 1.1.0
gensim >= 4.2.0
nltk >= 3.7.0
nbformat >= 5.7.0

# Optional (for BERT)
transformers >= 4.25.0
torch >= 1.13.0
accelerate >= 0.15.0
```

---

## 🔬 Technical Details

### Feature Engineering Pipeline

```
Raw Bengali Text
      │
      ▼
Text Normalization
(keep U+0980–U+09FF Bengali Unicode)
      │
      ▼
Tokenization → Stopword Removal
(custom 60-word Bengali stopword list)
      │
      ▼
Stemming / Lemmatization
(rule-based Bengali morphology)
      │
      ├──────────────────────────────────────┐
      ▼                                      ▼
TF-IDF Features                    Word Embeddings
(char 2-4 gram, 50K features)      (Word2Vec 100-dim)
      │                                      │
      ▼                                      ▼
Traditional ML Models          SVD(256) → DL Models
(NB, LR, SVM, RF)              (ANN, RNN, LSTM, GRU)
      │                                      │
      └──────────────┬───────────────────────┘
                     ▼
              Evaluation
        (Accuracy, F1, ROC, CM)
```

### Bengali NLP Challenges Addressed

| Challenge | Solution Used |
|-----------|--------------|
| No Bengali NLTK tokenizer | Custom whitespace tokenizer |
| Bengali inflection (ছেলেরা→ছেলে) | Rule-based stemmer + lemmatizer |
| No Bengali POS model | Heuristic suffix-based tagger |
| Bengali char encoding | Char TF-IDF on Unicode text |
| Class imbalance (2:1) | Complement Naive Bayes, stratified split |
| No Bengali word vectors | Word2Vec trained on corpus |

---

## 📊 Visualizations Generated

The notebook produces **25+ visualizations** including:

- 📊 Class & category distribution charts
- 🔥 Hate % per category heatmap
- 📈 Sentence length percentile plots
- 🌡️ TF-IDF weight heatmaps
- 🗺️ LSA 2D semantic space scatter plots
- 🧭 Word2Vec PCA with hate-ratio coloring
- 🌐 Bengali word co-occurrence matrix
- 🏗️ ANN/RNN/LSTM/GRU architecture diagrams
- 📉 DL training loss & accuracy curves
- 🎯 ROC curves + Precision-Recall curves
- 🔲 Confusion matrices for all 8 models
- 🕸️ Radar chart comparing top 5 models
- 🔍 4-head transformer attention heatmaps
- 🏆 Grand comparison bar chart + ranking table

---

## 🤗 BERT Fine-tuning

For best results, fine-tune Bengali BERT:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL = "sagorsarker/bangla-bert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

# Inference
def predict_hate(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax(-1).item()
    prob = torch.softmax(logits, dim=-1)[0][pred].item()
    return ("HATE" if pred == 1 else "NON-HATE"), f"{prob:.1%}"

# Example
text = "এই দেশের মানুষগুলো সত্যিই অনেক ভালো"
label, confidence = predict_hate(text)
print(f"Prediction: {label} ({confidence} confidence)")
```

---

## 📚 References

| Topic | Reference |
|-------|-----------|
| Bengali BERT | [BanglaBERT — sagorsarker](https://huggingface.co/sagorsarker/bangla-bert-base) |
| BERT | Devlin et al., 2018 — *BERT: Pre-training of Deep Bidirectional Transformers* |
| Word2Vec | Mikolov et al., 2013 — *Distributed Representations of Words and Phrases* |
| GloVe | Pennington et al., 2014 — *GloVe: Global Vectors for Word Representation* |
| TF-IDF | Salton & Buckley, 1988 — *Term-Weighting Approaches in Automatic Text Retrieval* |
| LSA | Deerwester et al., 1990 — *Indexing by Latent Semantic Analysis* |
| Transformer | Vaswani et al., 2017 — *Attention Is All You Need* |
| LSTM | Hochreiter & Schmidhuber, 1997 — *Long Short-Term Memory* |
| GRU | Cho et al., 2014 — *Learning Phrase Representations using RNN Encoder-Decoder* |
| Brill TBT | Brill, 1992 — *A Simple Rule-Based Part of Speech Tagger* |

---

## 🙏 Acknowledgements

- **Dataset** collected from Bengali social media platforms
- **Bengali BERT** model by [sagorsarker](https://huggingface.co/sagorsarker)
- **NLTK** for English NLP utilities and linguistic resources
- **Gensim** for Word2Vec implementation
- **Scikit-learn** for ML pipelines and evaluation metrics

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**BanglaGuard NLP Project**  
Developed as a comprehensive NLP research notebook covering 26 techniques  
applied on real Bengali hate speech data.

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

`Bengali NLP` • `Hate Speech Detection` • `Deep Learning` • `BERT` • `Word2Vec` • `TF-IDF`

</div>
