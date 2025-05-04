# Abbreviations-Integration-Sarcasm

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.8+](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-model NLP system for sarcasm detection with abbreviation handling, supporting:
- 🤖 Bi-LSTM/Bi-GRU with custom embedding layers
- 🌳 Random Forest with TF-IDF features
- 🧠 DistillBERT transformer model
- 📚 Integrated abbreviation dictionary from Kaggle

## Features

- **Abbreviation-Aware Processing**  
  Special handling for internet slang (LOL, SMH) using [Kaggle's Abbreviation Dataset](https://www.kaggle.com/datasets/ckapre51/common-abbreviations-and-short-forms-with-meanings)

## Installation 
1. Clone repository:
```bash
git clone https://github.com/TarunSamala/Abbreviations-Integration-Sarcasm.git