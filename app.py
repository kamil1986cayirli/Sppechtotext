import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from openai import OpenAI
from openai import RateLimitError, APITimeoutError, APIError

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Ses Dosyası → Türkçe Metin", layout="wide")
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
        'Streamlit Cloud → Manage app → Settings → Secrets:\n'
        'OPENAI_API_KEY = "sk-..."'
    )
    st.stop()

client = OpenAI(api_key=api_key)

# -------------------------------------------------
# SESSION LOCK (aynı anda 2 istek olmasın)
# -------------------------------------------------
if "busy" not in st.session_state:
    st.session_state.busy = False

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Ayarlar")
    model = st.selectbox(
        "Model",
        ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
        index=0
    )
    language = st.text_input("Dil", "tr")

    st.divider()
    max_mb = st.number_input("Maks. dosya boyutu (MB)", min_value=1, max_value=200, value=25, step=1)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Ses dosyası yükleyin",
    type=["wav", "mp3", "m4a", "mp4", "ogg", "flac", "webm"]
)

def transcribe_with_retry(file_path: Path, model: str, language: str, max_retries: int = 5):
    """
    RateLimit / geçici ağ hatalarında exponential backoff ile retry yapar.
    """
    base_sleep = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            return client.audio.transcriptions.create(
                file=file_path,
                model=model,
                language=language
            )
        except RateLimitError as e:
            # Exponential backoff + jitter
            sleep_s = base_sleep * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 30)  # üst limit
            st.warning(f"Rate limit aşıldı. {sleep_s:.0f} sn bekleyip tekrar deniyorum... (Deneme {attempt}/{max_retries})")
            time.sleep(sleep_s)
            last_err = e
        except (APITimeoutError, APIError) as e:
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 20)
            st.warning(f"Geçici API/ağ hatası. {sleep_s:.0f} sn sonra tekrar... (Deneme {attempt}/{max_retries})")
            time.sleep(sleep_s)
            last_err = e

    raise last_err  # son hatayı dışarı fırlat

if uploaded_file:
    st.audio(uploaded_file)

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > max_mb:
        st.error(f"Dosya çok büyük: {size_mb:.1f} MB. Limit: {max_mb} MB. Daha küçük dosya deneyin.")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.write("**Dosya Bilgisi**")
        st.json({"ad": uploaded_file.name, "boyut_mb": round(size_mb, 2), "tip": uploaded_file.type})

    with col2:
        btn = st.button("Transkribe Et", type="primary", disabled=st.session_state.busy)

    if btn:
        st.session_state.busy = True
        try:
            with st.spinner("Transkripsiyon yapılıyor..."):
                suffix = Path(uploaded_file.name).suffix or ".audio"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = Path(tmp.name)

                try:
                    result = transcribe_with_retry(tmp_path, model=model, language=language, max_retries=5)
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

            st.success("Transkripsiyon tamamlandı")
            st.text_area("Türkçe Metin", result.text, height=350)

            st.download_button(
                "TXT olarak indir",
                data=result.text.encode("utf-8"),
                file_name="transkript.txt",
                mime="text/plain"
            )

        except RateLimitError:
            st.error(
                "Rate limit / kota sınırına takıldınız.\n\n"
                "Çözüm: 1-2 dakika bekleyip tekrar deneyin; çok sık deneme yapmayın.\n"
                "Eğer sürekli oluyorsa: OpenAI projenizde billing/kota/limitleri kontrol edin."
            )
        except Exception as e:
            st.error(f"Beklenmeyen hata: {type(e).__name__}: {e}")
        finally:
            st.session_state.busy = False
