
import os
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="Ses Dosyası → Türkçe Metin (Ücretsiz)", layout="wide")
st.title("Ses Dosyası → Türkçe Metin (Ücretsiz)")
st.caption("Offline Whisper (faster-whisper) + Streamlit Cloud")

with st.sidebar:
    st.header("Ayarlar")
    model_size = st.selectbox("Model", ["tiny", "base", "small"], index=1)
    language = st.text_input("Dil", "tr")
    vad = st.toggle("VAD (sessizlik filtreleme)", value=True)
    max_mb = st.number_input("Maks. dosya boyutu (MB)", min_value=1, max_value=200, value=25, step=1)

@st.cache_resource
def load_model(size: str):
    return WhisperModel(size, device="cpu", compute_type="int8")

uploaded = st.file_uploader(
    "Ses dosyası yükleyin",
    type=["wav", "mp3", "m4a", "mp4", "ogg", "flac", "webm"]
)

if not uploaded:
    st.info("Dosya yükleyin, ardından 'Transkribe Et' butonuna basın.")
    st.stop()

st.audio(uploaded)

size_mb = uploaded.size / (1024 * 1024)
if size_mb > max_mb:
    st.error(f"Dosya çok büyük: {size_mb:.1f} MB. Limit: {max_mb} MB.")
    st.stop()

st.write("**Dosya Bilgisi**")
st.json({"ad": uploaded.name, "boyut_mb": round(size_mb, 2), "tip": uploaded.type})

model = load_model(model_size)

if st.button("Transkribe Et", type="primary"):
    with st.spinner("Model çalışıyor (ilk seferde model indirilebilir)..."):
        suffix = Path(uploaded.name).suffix or ".audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        try:
            segments, info = model.transcribe(
                tmp_path,
                language=language,
                vad_filter=vad,
                vad_parameters=dict(min_silence_duration_ms=500) if vad else None,
                beam_size=5,
            )

            lines = [seg.text.strip() for seg in segments if seg.text.strip()]
            text = "\n".join(lines)

            st.success(f"Tamamlandı. Dil: {info.language} (p={info.language_probability:.2f})")
            st.text_area("Türkçe Metin", text, height=350)

            st.download_button(
                "TXT indir",
                data=text.encode("utf-8"),
                file_name="transkript.txt",
                mime="text/plain"
            )

        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
