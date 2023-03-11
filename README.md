# Sentiment Analysis Using Encoder-Based Transformer from SCRATCH
___
The project is mainly focused on building Transformer-Based Classifier from scratch. For simplicity, I only used Encoder Blocks of transformer in the model. To ensure that model works, I trained it on a simple Tweets dataset from Kaggle. There is link below for a simple Streamlit-Based API for the model.
* [Application Link](https://lukabarbakadze-sentiment-analysis-using-transformer-app-uccs6l.streamlit.app/)
* [Dataset Link (Kaggle)](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
___
### Files Description
* main.ipynb - Main Working File
* app.py - Simple Application hosted on Streamlit Cloud to access model
* Model_Builder.py - Module which defines Transformer Bult in main.ipynb
* word2int.py - simple word-to-integer encoder
___
### Table of Contents (main.py)
* Imports
* Upload Data
* First Look
* Drop Missing Values
* Explanatory Data Analysis
  * Check Target Variable Distribution
  * Check "Irrelevant" Class
  * Drop "Irrelevant" Class
  * Check Length of Tokenized Tweets
* Data Preprocess
  * Tokenize Tweets
  * Encode Tokens
  * Generate X & y Matrices
  * Define Dataset & DataLoaders for Batch Processing
* Model Building
  * Model Parameters
  * Device
  * Embadding Class
  * Define Signle Self-Attention Head
  * Define Multi-Head Self-Attention
  * FeedForward Layer for Encoder Block
  * Build Single Transformer Encoder Block
  * Define FinalLayer for Classification
  * Build TransformerClassifier
  * Define Model & Check Weights
* Training
  * Define loss & optimizer
  * Trainig Loop
* Visualize Results
  * Save Model Weights & Vocabulary
___
### Acknowledgements
* [Paper of Transformer Nueral Network](https://arxiv.org/abs/1706.03762)
* [Andrej Karpathy's Youtube Video about chatGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=4876s&ab_channel=AndrejKarpathy)
___
