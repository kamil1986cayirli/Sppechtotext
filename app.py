import os
import time
import tempfile
import hashlib
from pathlib import Path

import streamlit as st
from openai import OpenAI
from openai import RateLimitError, APITimeoutError, APIError

st.set_page_config(page_title="Ses Dosyası → Türkçe Metin", layout="wide")
st.title("Ses Dosyası → Türkçe Metin")
st.caption("OpenAI Speech-to-Text API + Streamlit Cloud")

# ----------------------------
# API KEY
# ----------------------------
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error('OPENAI_API_KEY yok. Streamlit Cloud → Secrets:  OPENAI_API_KEY = "sk-..."')
    st.stop()

client = OpenAI(api_key=api_key)

# ----------------------------
# Session state
# ----------------------------
st.session_state.setdefault("busy", False)
st.session_state.setdefault("last_file_sig", None)
st.session_state.setdefault("last_transcript", "")

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
    st.caption("Not: Rate limit/kota durumunda uygulama çökmez; uyarı verir.")

uploaded_file = st.file_uploader(
    "Ses dosyası yükleyin",
    type=["wav", "mp3", "m4a", "mp4", "ogg", "flac", "webm"],
)

def file_signature(file_obj) -> str:
    head = file_obj.getbuffer()[:1024 * 1024]
    h = hashlib.sha256()
    h.update(head)
    h.update(str(file_obj.size).encode("utf-8"))
    h.update(file_obj.name.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def retry_after_seconds(err: RateLimitError) -> float | None:
    """
    OpenAI SDK hata objesinde Retry-After bilgisi gelirse onu okumaya çalışır.
    Farklı SDK versiyonlarında header erişimi değişebildiği için güvenli yazıldı.
    """
    try:
        # err.response headers (bazı sürümlerde)
        headers = getattr(err, "response", None)
        if headers is not None and hasattr(headers, "headers"):
            ra = headers.headers.get("retry-after") or headers.headers.get("Retry-After")
            if ra:
                return float(ra)
    except Exception:
        pass
    return None

def transcribe_with_retry(file_path: Path, max_retries: int = 6) -> str:
    base_sleep = 2.0
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            res = client.audio.transcriptions.create(
                file=file_path,
                model=model,
                language=language
            )
            return res.text

        except RateLimitError as e:
            last_err = e
            ra = retry_after_seconds(e)
            sleep_s = ra if ra is not None else min(base_sleep * (2 ** (attempt - 1)), 60)

            st.warning(
                f"Rate limit/kota. {sleep_s:.0f} sn bekleyip tekrar deniyorum... "
                f"({attempt}/{max_retries})"
            )
            time.sleep(sleep_s)

        except (APITimeoutError, APIError) as e:
            last_err = e
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 30)
            st.warning(f"Geçici API/ağ hatası. {sleep_s:.0f} sn sonra tekrar... ({attempt}/{max_retries})")
            time.sleep(sleep_s)

    # Buraya geldiysek başarısız.
    raise last_err

if not uploaded_file:
    st.info("Dosya yükleyin. Otomatik açık ise yükleme sonrası işlem başlar.")
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

should_auto_run = auto_run and (st.session_state.last_file_sig != sig)

def run_transcription():
    st.session_state.busy = True
    try:
        suffix = Path(uploaded_file.name).suffix or ".audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        try:
            text = transcribe_with_retry(tmp_path, max_retries=6)
            st.session_state.last_transcript = text
            st.success("Transkripsiyon tamamlandı.")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except RateLimitError:
        st.error(
            "Rate limit veya kota/billing sorunu.\n\n"
            "Yapılacaklar:\n"
            "1) 1-2 dakika bekleyip tekrar deneyin.\n"
            "2) OpenAI hesabınızda ilgili Project/Org için Billing ve Usage limitlerini kontrol edin.\n"
            "3) Aynı anda çok deneme yapmayın (Streamlit rerun + tekrar tıklama limiti tetikler)."
        )
    except Exception as e:
        st.error(f"Beklenmeyen hata: {type(e).__name__}: {e}")
    finally:
        st.session_state.busy = False

if manual or should_auto_run:
    st.session_state.last_file_sig = sig
    with st.spinner("Transkripsiyon yapılıyor..."):
        run_transcription()

if st.session_state.last_transcript:
    st.text_area("Türkçe Metin", st.session_state.last_transcript, height=350)
    st.download_button(
        "TXT olarak indir",
        data=st.session_state.last_transcript.encode("utf-8"),
        file_name="transkript.txt",
        mime="text/plain"
    )
else:
    st.info("Transkripsiyon henüz üretilmedi. Otomatik kapalıysa butona basın.")
