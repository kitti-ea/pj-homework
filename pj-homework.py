import streamlit as st
import pandas as pd
import openai
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function to create word cloud
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Function to plot word cloud
def plot_wordcloud(wordcloud):
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Sidebar for API Key input
st.sidebar.title("OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API key here", type="password")
openai.api_key = api_key

# File uploader
uploaded_file = st.file_uploader("Upload your CSV, Excel, or TXT file", type=["csv", "xlsx", "txt"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        
        st.write("File content:")
        st.dataframe(df.head())

        # Concatenate all text from the dataframe
        all_text = ' '.join(df.astype(str).values.flatten())

        # Send to OpenAI ChatGPT API
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=f"Find the most frequent words in the following text and create a word cloud: {all_text}",
            max_tokens=100
        )
        word_freq = response.choices[0].text.strip()

        # Create word cloud
        wordcloud = create_wordcloud(word_freq)

        # Display word cloud
        plot_wordcloud(wordcloud)
        
        # Provide option to download word cloud image
        st.sidebar.download_button(
            label="Download Word Cloud",
            data=wordcloud.to_image().tobytes(),
            file_name="wordcloud.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
