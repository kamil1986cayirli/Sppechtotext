
import os
import tempfile
from pathlib import Path
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Ses → Metin (TR)", layout="wide")
st.title("Ses Dosyası → Türkçe Metin")
st.caption("OpenAI Speech-to-Text API + Streamlit Cloud")

client = OpenAI()

with st.sidebar:
    st.header("Ayarlar")
    model = st.selectbox(
        "Model",
        ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
        index=0
    )
    language = st.text_input("Dil", "tr")
    st.info("OPENAI_API_KEY Streamlit Secrets içine eklenmelidir.")

uploaded = st.file_uploader(
    "Ses dosyası yükleyin",
    type=["wav", "mp3", "m4a", "mp4", "ogg", "flac", "webm"]
)

if uploaded:
    st.audio(uploaded)
    if st.button("Transkribe Et", type="primary"):
        with st.spinner("Transkripsiyon yapılıyor..."):
            suffix = Path(uploaded.name).suffix or ".audio"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name

            try:
                result = client.audio.transcriptions.create(
                    file=Path(tmp_path),
                    model=model,
                    language=language
                )
                st.text_area("Transkript", result.text, height=350)
            finally:
                os.remove(tmp_path)
