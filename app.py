import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="Ses Dosyası → Türkçe Metin (Ücretsiz)", layout="wide")
st.title("Ses Dosyası → Türkçe Metin (Ücretsiz)")
st.caption("Offline Whisper (faster-whisper) + Streamlit Cloud")

with st.sidebar:
    st.header("Ayarlar")
    model_size = st.selectbox("Model", ["tiny", "base", "small"], index=2)  # small default
    language = st.text_input("Dil", "tr")

    # Doğruluk için float32 daha iyi (daha yavaş), int8 daha hızlı (daha hatalı olabilir)
    compute_type = st.selectbox("Compute type", ["int8", "float32"], index=1)

    vad = st.toggle("VAD (sessizlik filtreleme)", value=False)  # TTS için kapalı önerilir
    beam_size = st.slider("Beam size (doğruluk ↑ / hız ↓)", min_value=1, max_value=10, value=8)
    max_mb = st.number_input("Maks. dosya boyutu (MB)", min_value=1, max_value=200, value=25, step=1)

@st.cache_resource
def load_model(size: str, compute: str):
    return WhisperModel(size, device="cpu", compute_type=compute)

def to_wav_16k_mono(src_path: Path) -> Path:
    """
    Her formatı 16kHz mono PCM WAV'a çevirir (Whisper için en stabil giriş).
    """
    dst = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return dst

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

model = load_model(model_size, compute_type)

if st.button("Transkribe Et", type="primary"):
    with st.spinner("Ses hazırlanıyor ve transkribe ediliyor..."):
        suffix = Path(uploaded.name).suffix or ".audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_src = Path(tmp.name)

        tmp_wav = None
        try:
            # 1) Format normalize
            tmp_wav = to_wav_16k_mono(tmp_src)

            # 2) Transcribe
            segments, info = model.transcribe(
                str(tmp_wav),
                language=language,
                vad_filter=vad,
                vad_parameters=dict(min_silence_duration_ms=500) if vad else None,
                beam_size=beam_size,
            )

            lines = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
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
            for p in [tmp_src, tmp_wav]:
                try:
                    if p and Path(p).exists():
                        os.remove(p)
                except Exception:
                    pass
