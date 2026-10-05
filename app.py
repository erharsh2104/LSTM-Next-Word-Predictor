import streamlit as st
import pickle 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model('Next_word_pred.h5') 

with open('tokenizer.pkl','rb') as file :
    tokenizer=pickle.load(file)

reverse_index={idx:word for word,idx in tokenizer.word_index.items()}

max_len=166

def generate_text(seed_text,num_words=10):
    text=seed_text
    for _ in range(num_words):
        sent=tokenizer.texts_to_sequences([text])[0]
        padded=pad_sequences([sent],maxlen=max_len,padding='pre')
        preds=model.predict(padded,verbose=0)
        pos=np.argmax(preds)
        next_word= reverse_index.get(pos,' ')
        text += " " + next_word
    return text


st.title("Next Word Prediction With Deep Learning")

seed=st.text_input('Enter the Starting text:','Hello')

num_words=st.slider("Number of Words to Generate",1,20,10)

if st.button("Generate"):
    results=generate_text(seed,num_words)
    st.write("Result:", results)