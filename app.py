import streamlit as st
from generate import generate

st.title("Finance Risk LLM")

input_text = st.text_input("Enter your financial prompt:")

if st.button("Generate"):
    result = generate(model, tokenizer, input_text)
    st.write(result)
