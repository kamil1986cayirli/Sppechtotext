import os
import time
import tempfile
import hashlib
from pathlib import Path

import streamlit as st
from openai import OpenAI
from openai import RateLimitError, APITimeoutError, APIError

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Ses Dosyası → Türkçe Metin", layout="wide")
st.title("Ses Dosyası → Türkçe Metin")
st.caption("OpenAI Speech-to-Text API + Streamlit Cloud")

# ----------------------------
# API KEY
# ----------------------------
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error('OPENAI_API_KEY bulunamadı. Secrets’a şu satırı ekleyin:  OPENAI_API_KEY = "sk-..."')
    st.stop()

client = OpenAI(api_key=api_key)

# ----------------------------
# Session state
# ----------------------------
if "busy" not in st.session_state:
    st.session_state.busy = False
if "last_file_sig" not in st.session_state:
    st.session_state.last_file_sig = None
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Ayarlar")
    model = st.selectbox(
        "Model",
        ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
        index=0
    )
    language = st.text_input("Dil", "tr")
    max_mb = st.number_input("Maks. dosya boyutu (MB)", min_value=1, max_value=200, value=25, step=1)
    auto_run = st.toggle("Dosya yüklenince otomatik transkribe et", value=True)

# ----------------------------
# Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "Ses dosyası yükleyin",
    type=["wav", "mp3", "m4a", "mp4", "ogg", "flac", "webm"],
    key="audio_uploader"
)

def file_signature(file_obj) -> str:
    """
    Streamlit rerun'larında aynı dosyayı tekrar işlememek için imza üretir.
    Büyük dosyayı komple hashlemiyoruz; ilk 1MB + name + size yeterli.
    """
    head = file_obj.getbuffer()[:1024 * 1024]
    h = hashlib.sha256()
    h.update(head)
    h.update(str(file_obj.size).encode("utf-8"))
    h.update(file_obj.name.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def transcribe_with_retry(file_path: Path, max_retries: int = 5):
    base_sleep = 2.0
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.audio.transcriptions.create(
                file=file_path,
                model=model,
                language=language
            )
        except RateLimitError as e:
            last_err = e
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 30)
            st.warning(f"Rate limit. {sleep_s:.0f} sn bekleyip tekrar deniyorum... ({attempt}/{max_retries})")
            time.sleep(sleep_s)
        except (APITimeoutError, APIError) as e:
            last_err = e
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 20)
            st.warning(f"Geçici hata. {sleep_s:.0f} sn sonra tekrar... ({attempt}/{max_retries})")
            time.sleep(sleep_s)
    raise last_err

def run_transcription():
    st.session_state.busy = True
    try:
        suffix = Path(uploaded_file.name).suffix or ".audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        try:
            result = transcribe_with_retry(tmp_path, max_retries=5)
            st.session_state.last_transcript = result.text
            st.success("Transkripsiyon tamamlandı.")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    finally:
        st.session_state.busy = False

# ----------------------------
# Main flow
# ----------------------------
if not uploaded_file:
    st.info("Dosya yükleyin. Otomatik transkribe açık ise yükleme sonrası işlem başlayacak.")
    st.stop()

st.audio(uploaded_file)

size_mb = uploaded_file.size / (1024 * 1024)
if size_mb > max_mb:
    st.error(f"Dosya çok büyük: {size_mb:.1f} MB. Limit: {max_mb} MB.")
    st.stop()

sig = file_signature(uploaded_file)

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    st.write("**Dosya Bilgisi**")
    st.json({"ad": uploaded_file.name, "boyut_mb": round(size_mb, 2), "tip": uploaded_file.type})

with col2:
    manual = st.button("Transkribe Et", type="primary", disabled=st.session_state.busy)

# Otomatik çalıştırma: yeni dosya yüklendiyse ve auto açık ise
should_auto_run = auto_run and (st.session_state.last_file_sig != sig)

if manual or should_auto_run:
    st.session_state.last_file_sig = sig
    with st.spinner("Transkripsiyon yapılıyor..."):
        run_transcription()

# Çıktı
if st.session_state.last_transcript:
    st.text_area("Türkçe Metin", st.session_state.last_transcript, height=350)
    st.download_button(
        "TXT olarak indir",
        data=st.session_state.last_transcript.encode("utf-8"),
        file_name="transkript.txt",
        mime="text/plain"
    )
else:
    st.info("Henüz transkripsiyon çalışmadı. Otomatik kapalıysa 'Transkribe Et' butonuna basın.")
