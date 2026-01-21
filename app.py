import os
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAI

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Ses Dosyası → Türkçe Metin",
    layout="wide"
)

st.title("Ses Dosyası → Türkçe Metin")
st.caption("OpenAI Speech-to-Text API + Streamlit Cloud")

# -------------------------------------------------
# API KEY (SADECE SECRETS / ENV)
# -------------------------------------------------
api_key = None

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
elif os.getenv("OPENAI_API_KEY"):
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "OPENAI_API_KEY bulunamadı.\n\n"
        "Streamlit Cloud → Manage app → Settings → Secrets bölümüne:\n\n"
        'OPENAI_API_KEY = "sk-..."\n\n'
        "ekleyin."
    )
    st.stop()

client = OpenAI(api_key=api_key)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Ayarlar")

    model = st.selectbox(
        "Model",
        [
            "gpt-4o-mini-transcribe",  # önerilen
            "gpt-4o-transcribe",
            "whisper-1"
        ],
        index=0
    )

    language = st.text_input("Dil", "tr")

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Ses dosyası yükleyin",
    type=["wav", "mp3", "m4a", "mp4", "ogg", "flac", "webm"]
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Transkribe Et", type="primary"):
        with st.spinner("Transkripsiyon yapılıyor..."):
            suffix = Path(uploaded_file.name).suffix or ".audio"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                result = client.audio.transcriptions.create(
                    file=Path(tmp_path),
                    model=model,
                    language=language
                )

                st.success("Transkripsiyon tamamlandı")

                st.text_area(
                    "Türkçe Metin",
                    result.text,
                    height=350
                )

                st.download_button(
                    "TXT olarak indir",
                    data=result.text.encode("utf-8"),
                    file_name="transkript.txt",
                    mime="text/plain"
                )

            finally:
                os.remove(tmp_path)
