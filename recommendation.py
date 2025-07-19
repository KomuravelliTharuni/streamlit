# recommendation_app.py
# Program: recommendation_app.py
# Real-Time Product Recommendation Engine using Streamlit and Scikit-learn
# This application allows users to select a product and receive real-time recommendations based on product descriptions.
#  
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Sample Dataset --------------------
@st.cache_data
def load_data():
    return pd.DataFrame({
        'Product': [
            "iPhone 13 Pro Max", "Samsung Galaxy S22 Ultra", "Google Pixel 7",
            "OnePlus 11", "Xiaomi Mi 11 Ultra", "Apple MacBook Air M2",
            "Dell XPS 13", "Asus ROG Gaming Laptop", "iPad Pro 12.9", "Samsung Galaxy Tab S8"
        ],
        'Description': [
            "Apple smartphone with A15 Bionic chip, ProMotion display",
            "Samsung phone with great camera, 100x zoom, S Pen support",
            "Google Pixel with Tensor chip, best Android OS integration",
            "Flagship OnePlus phone with Snapdragon processor and fast charging",
            "Xiaomi high-end camera phone with AMOLED screen",
            "Apple laptop with M2 chip, thin and lightweight design",
            "Windows laptop with premium build and good performance",
            "High-performance gaming laptop with RGB and cooling",
            "Apple tablet with M1 chip and 120Hz display",
            "Samsung tablet with S Pen and DeX mode"
        ]
    })

# -------------------- Recommendation Function --------------------
def get_recommendations(selected_product, df, top_n=3):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = df[df['Product'] == selected_product].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the product itself
    recommendations = [df['Product'].iloc[i[0]] for i in sim_scores]
    return recommendations

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="SmartCart AI", layout="centered")
st.title("Real-Time Commerce Advisor")
st.caption("Select a product and see similar recommendations instantly.")

# Load Data
df = load_data()

# User Selection
product_selected = st.selectbox("Choose a Product", df['Product'].tolist())

if product_selected:
    st.markdown(f"### You selected: {product_selected}")
    recommended = get_recommendations(product_selected, df)
    
    st.markdown("### Recommended Products:")
    for i, rec in enumerate(recommended, start=1):
        st.write(f"{i}. {rec}")

# Show entire product list for reference
with st.expander("View Product Descriptions"):
    st.dataframe(df)