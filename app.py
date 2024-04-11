import streamlit as st
import torch
import numpy as np
import tiktoken

from utilities.dataloader import text_to_token_ids, token_ids_to_text
from model.transformer import TransformerModel
from generate import generate 

CONFIG = {
    "vocab_size": 50257,  
    "ctx_len": 1024,      
    "emb_dim": 768,       
    "n_heads": 12,        
    "n_layers": 12,       
    "drop_rate": 0,       
    "qkv_bias": False    
    }

torch.manual_seed(123)
model = TransformerModel(CONFIG)
model.load_state_dict(torch.load("model.pth"))
model.eval()  

tokenizer = tiktoken.get_encoding("gpt2")

@st.cache_data  # Cache results to improve performance
def run_generation(input_text, max_length=50, temperature=1.0, top_k=10):
    encoded = text_to_token_ids(input_text, tokenizer)
    
    out = generate(
        model=model,
        idx=encoded,
        max_new_tokens=max_length,
        context_size=CONFIG["ctx_len"],
        top_k=top_k,
        temperature=temperature
    )

    generated_text = token_ids_to_text(out, tokenizer).strip()
    return generated_text

# Set up title and sidebar options
st.title('Story Generator')

start_context = st.text_area('Start Context', value="Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine.One day,", height=150)

num_words = st.slider('Max New Words:', min_value=10, max_value=200, step=5, value=50)
temperature = st.slider('Temperature:', min_value=0.0, max_value=2.0, step=0.1, value=1.0)
top_k = st.slider('Top K Sampling:', min_value=0, max_value=50, step=5, value=10)

if st.button('Generate'):
    output_text = run_generation(start_context, max_length=num_words, temperature=temperature, top_k=top_k)
    st.write(f"\nGenerated Text:\n{output_text}")