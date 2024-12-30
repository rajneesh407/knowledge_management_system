import streamlit as st
from gtts import gTTS
from io import BytesIO

# Title
st.title("Text-to-Speech with Play Button")

# Text input
mytext = st.text_area(
    "Enter the text you want to convert to speech:",
    value="""Attention in this context refers to an attention function, which is a mechanism that maps a query and a set of key-value pairs to an output. The input consists of queries and keys of dimension dk, and values of dimension dv. The attention function computes the dot products of the queries with all keys, applies a softmax function to obtain the weights on the query with all keys, divides each by √ dk, and then multiplies the values with these weights. This process is also known as Scaled Dot-Product Attention.""",
)

if st.button("Play"):
    language = 'en'
    tts = gTTS(text=mytext, lang=language, slow=False)
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)  # Rewind to the beginning
    st.audio(audio_stream, format="audio/mp3")
