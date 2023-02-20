import torch
from torch import nn
import numpy as np
import Model_Builder
import nltk
from nltk.tokenize import word_tokenize
import Model_Builder
import word2int
nltk.download('punkt')
import streamlit as st

# hide hamburger
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# define word-to-integer encoder and main transformer encoder-only based model
Word2intEncoder = word2int.WordToInteger2()
model = Model_Builder.TransformerClassifier()

# define index to class dictionary
idx2class = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}

# title of App
st.title('Detect Sentiment of Tweet')
# get text/tweet from user
with st.form("Tweet", clear_on_submit = True):
    msg = st.text_area("Your Tweet: ", value = "")
    submit_button = st.form_submit_button(label="Send")
# get embadding of tweet
emb = Word2intEncoder(msg)
# Prepare Tensor embedding for model
emb = torch.Tensor(emb).unsqueeze(dim=0).long()
# get prediction/tweet class
clas = int(torch.argmax(model(emb).detach(), axis=1))
# get corresponding class label
pred = idx2class[clas]
# add corresponding emoji to class
emoji = ["blush", "neutral_face", "slightly_frowning_face"]
# return Tweet Sentiment to User
if msg == True:
    st.write(f"Tweet Sentiment: {pred} :{emoji[clas]}:")