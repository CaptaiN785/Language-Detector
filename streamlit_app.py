import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle
import warnings
warnings.filterwarnings("ignore")

##  -- Configurations and functions

st.set_page_config(
    page_title="Language Detector",
    page_icon="📙",
    layout="wide"
)

model = pickle.load(open("model.pkl", "rb"))
vec = pickle.load(open("vectorizer.pkl", "rb"))
df = pd.read_csv("data.csv")

def predict(sentence):
    sentence = np.array([sentence])    
    test = vec.transform(sentence)
    pred = model.predict(test)
    return pred[0]

### Configuration ends --------------


## Page begins.

st.title("Language Detector!!")

## Displaying all languages supported.
with st.container():
    languages = sorted(df['language'].unique(), key=lambda x : len(x))
    
    lang_iter = iter(languages)
    for col in st.columns(11):
        col.info(next(lang_iter))
    
    for col in st.columns(10):
        col.info(next(lang_iter))

col1, col2 = st.columns([7, 3])
example = ""

lang = col2.selectbox( "Select a example input", options=["Select any"] + languages)
if lang and lang != "Select any":
    example = df[df["language"] == lang]['Text'].sample(1).values[0]

sentence = col1.text_input("Enter a sentence of a languages..", \
    value=example, placeholder="Enter a sentence of any languages above.")

if sentence:
    st.success("Predicted language is : ***" + predict(sentence)+"***")