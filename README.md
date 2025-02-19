# NLP Abbreviation and Long-Form Detection 

##  Project Overview
This project is part of the **Natural Language Processing (COMM061) coursework** and focuses on **detecting abbreviations and their corresponding long forms** using deep learning techniques. The goal is to develop a sequence classification model utilizing **BiLSTM and CRF** to improve accuracy in identifying abbreviations in text.

## Dataset
The **PLOD-CW** dataset is an English-language dataset containing abbreviations and their corresponding long forms, collected from PLOS journals. It is designed for **Natural Language Processing (NLP) research in abbreviation detection**, specifically within the **scientific domain**.

[PLOD-CW Dataset on Hugging Face](https://huggingface.co/datasets/surrey-nlp/PLOD-CW)

## Features
- Implements **abbreviation and long-form detection**.
- Uses **BiLSTM with 4 layers and a hidden dimension of 256**.
- Compares **Cross-Entropy Loss and CRF (Negative Log-Likelihood)**.
- Evaluates **different word embedding techniques (GloVe & FastText)**.
- Analyzes **hyperparameters (batch size, learning rate, dropout, number of layers)**.

## Model Architecture & Configuration
- **Model:** BiLSTM with 4 layers and 256 hidden dimensions.
- **Loss Function:** Comparison between **Cross-Entropy Loss** and **Negative Log-Likelihood (CRF)**.
- **Word Embeddings:** Evaluating performance differences between **GloVe** and **FastText**.
- **Hyperparameter Tuning:** Optimized **batch size, learning rate, dropout value, and number of layers** for best performance.

## 📊 Experimental Results

### **1. Word Embeddings Comparison**
- **GloVe** (200K vocab, 100D): Performed better for abbreviation detection.
- **FastText** (200K vocab, 100D): Better at handling OOV words but had lower recall.

### **2. Loss Function Comparison**
- **Cross-Entropy Loss**: Showed signs of overfitting.
- **Negative Log-Likelihood (CRF)**: Performed better due to sequence-level dependencies.

### **3. Hyperparameter Optimization**
| Hyperparameter       | Best Value | Notes |
|----------------------|-----------|------------------------------|
| **Learning Rate**    | `0.001`   | Balanced convergence and generalization. |
| **Dropout Value**    | `0.4`     | Higher values led to poor performance. |
| **Batch Size**       | `16`      | Best accuracy; `32` and `64` were also viable. |
| **Hidden Dimension** | `256`     | Larger values caused overfitting. |
| **Number of Layers** | `4`       | Optimal depth for performance. |


