import streamlit as st
import torch
import pickle
from model import MiniFinanceLLM

st.set_page_config(page_title="Finance Credit Risk LLM", layout="centered")

st.title("📊 Finance Credit Risk LLM")
st.markdown("Domain-Specific Language Model for Credit Risk & Financial Insights")

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

word_to_idx, idx_to_word = load_tokenizer()

# Load model
@st.cache_resource
def load_model():
    model = MiniFinanceLLM(len(word_to_idx))
    model.load_state_dict(torch.load("finance_llm.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

sequence_length = 4

def generate_text(start_text, length=20):
    words_input = start_text.lower().split()
    
    input_seq = [word_to_idx.get(word, 0) for word in words_input[-sequence_length:]]
    
    for _ in range(length):
        x = torch.tensor([input_seq])
        with torch.no_grad():
            output = model(x)
        
        predicted = torch.argmax(output, dim=1).item()
        words_input.append(idx_to_word[predicted])
        input_seq = input_seq[1:] + [predicted]
    
    return " ".join(words_input)

user_input = st.text_input("Enter financial query:")

if st.button("Generate Insight"):
    if user_input.strip() == "":
        st.warning("Please enter a financial query.")
    else:
        result = generate_text(user_input, 25)
        st.success("Generated Insight:")
        st.write(result)

st.sidebar.header("Model Info")
st.sidebar.write("• Custom Tokenizer")
st.sidebar.write("• Mini Transformer Architecture")
st.sidebar.write("• Trained on Domain-Specific Credit Risk Corpus")
st.sidebar.write("• Built from Scratch using PyTorch")
