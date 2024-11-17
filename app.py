import streamlit as st
import pickle


model=pickle.load(open('model.pkl','rb'))
tfidf=pickle.load(open('tfidf_vectorizer.pkl','rb'))

st.title("Emotion Analysis Using Text")

input_sms=st.text_area("Enter your thoughts,here.")
emotions=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

if st.button('Predict') and input_sms!="":
    #1.Vectorize
    vectors=tfidf.transform([input_sms])
    #2.Predict
    result=emotions[model.predict(vectors)[0]]
    #3.Display
    st.header('You are feeling, '+' '+result)