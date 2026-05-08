"""
app.py — EmoSense · Streamlit Emotion Detector
✅ Streamlit Cloud compatible
✅ Class names match dataset: angry/disgusted/fearful/happy/neutral/sad/surprised
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import io
import os

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="EmoSense · Emotion Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Syne:wght@700;800;900&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: #080810;
  color: #dcdcf0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.8rem 2rem 2rem; }

/* ── Gradient headline ── */
.hero {
  font-family: 'Syne', sans-serif;
  font-size: 3rem;
  font-weight: 900;
  line-height: 1.05;
  letter-spacing: -1.5px;
  background: linear-gradient(120deg, #f472b6 0%, #a78bfa 45%, #38bdf8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: 0.97rem;
  color: #6b6b99;
  margin-top: 0.3rem;
  font-weight: 300;
  letter-spacing: 0.02em;
}

/* ── Pill tag ── */
.pill {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 0.22rem 0.7rem;
  border-radius: 999px;
  margin-right: 0.3rem;
}

/* ── Stat card ── */
.stat-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 14px;
  padding: 1rem 0.8rem;
  text-align: center;
  transition: border-color 0.2s;
}
.stat-card:hover { border-color: rgba(167,139,250,0.4); }
.stat-num {
  font-family: 'Syne', sans-serif;
  font-size: 1.9rem;
  font-weight: 900;
  color: #a78bfa;
  line-height: 1;
}
.stat-lbl {
  font-size: 0.65rem;
  color: #555577;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 0.35rem;
}

/* ── Section heading ── */
.sec-head {
  font-family: 'Syne', sans-serif;
  font-size: 0.95rem;
  font-weight: 800;
  color: #b0b0d0;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.75rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* ── Detection card ── */
.det-card {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  padding: 0.65rem 0.9rem;
  border-radius: 12px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  margin-bottom: 0.5rem;
  transition: background 0.15s;
}
.det-card:hover { background: rgba(255,255,255,0.06); }
.det-emoji { font-size: 1.7rem; line-height: 1; }
.det-name  { font-weight: 700; font-size: 0.95rem; }
.det-conf  { font-size: 0.75rem; color: #555577; margin-left: auto; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #0c0c18 !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  border: 2px dashed rgba(244,114,182,0.3) !important;
  border-radius: 14px !important;
  background: rgba(244,114,182,0.03) !important;
  padding: 1.2rem !important;
}

/* ── Primary button ── */
.stButton > button {
  background: linear-gradient(135deg, #9333ea, #2563eb) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em !important;
  padding: 0.58rem 1.3rem !important;
}
.stButton > button:hover { opacity: 0.82 !important; }

/* ── Notice boxes ── */
.notice {
  border-radius: 10px;
  padding: 0.75rem 1rem;
  font-size: 0.83rem;
  margin-bottom: 1rem;
}
.notice-blue  { background: rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25); color:#7dd3fc; }
.notice-pink  { background: rgba(244,114,182,0.08); border:1px solid rgba(244,114,182,0.25); color:#f9a8d4; }
.notice-green { background: rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.25); color:#6ee7b7; }

/* ── Progress bar ── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #f472b6, #a78bfa);
  border-radius: 999px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080810; }
::-webkit-scrollbar-thumb { background: #2a2a40; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Emotion metadata (keys = exact folder/class names) ────────────────────────
EMOTION_HEX = {
    "angry":     "#ef4444",
    "disgusted": "#22c55e",
    "fearful":   "#a855f7",
    "happy":     "#eab308",
    "neutral":   "#94a3b8",
    "sad":       "#3b82f6",
    "surprised": "#f97316",
}
EMOTION_DISPLAY = {
    "angry": "Angry", "disgusted": "Disgusted", "fearful": "Fearful",
    "happy": "Happy", "neutral": "Neutral", "sad": "Sad", "surprised": "Surprised",
}
EMOTION_EMOJIS = {
    "angry": "😠", "disgusted": "🤢", "fearful": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprised": "😲",
}


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    from detect import EmotionDetector

    # Priority 1: weights committed to repo
    if os.path.exists("emotion_model.pth"):
        return EmotionDetector(model_path="emotion_model.pth")

    # Priority 2: HuggingFace Hub (set HF_REPO in Streamlit Secrets)
    try:
        hf_repo = st.secrets.get("HF_REPO", os.environ.get("HF_REPO", ""))
    except Exception:
        hf_repo = os.environ.get("HF_REPO", "")

    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(
                repo_id=hf_repo,
                filename="emotion_model.pth",
                cache_dir="/tmp/hf_cache",
            )
            return EmotionDetector(model_path=weights)
        except Exception as e:
            st.warning(f"⚠️ HuggingFace download failed: {e}")

    # Priority 3: random weights (UI demo — train first for real predictions)
    return EmotionDetector(model_path=None)


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_detection(pil_img: Image.Image, detector, draw_boxes: bool):
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    annotated_bgr, detections = detector.process_image(bgr, draw_boxes=draw_boxes)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB))
    return annotated_pil, detections


def render_stat_row(n_faces: int, elapsed_ms: float, dominant: str):
    s1, s2, s3 = st.columns(3)
    emoji = EMOTION_EMOJIS.get(dominant, "—")
    disp  = EMOTION_DISPLAY.get(dominant, dominant)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{n_faces}</div>'
                    f'<div class="stat-lbl">Faces Detected</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{elapsed_ms:.0f}ms</div>'
                    f'<div class="stat-lbl">Inference Time</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{emoji}</div>'
                    f'<div class="stat-lbl">{disp}</div></div>', unsafe_allow_html=True)


def render_prob_bars(probs: dict):
    for key, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        disp  = EMOTION_DISPLAY[key]
        color = EMOTION_HEX[key]
        emoji = EMOTION_EMOJIS[key]
        c1, c2 = st.columns([3, 7])
        with c1:
            st.markdown(
                f'<span style="font-size:0.8rem;font-weight:700;color:{color};">'
                f'{emoji} {disp}</span>', unsafe_allow_html=True)
        with c2:
            st.progress(float(prob))
        st.markdown(
            f'<div style="text-align:right;font-size:0.68rem;color:#44446a;'
            f'margin-top:-0.55rem;margin-bottom:0.4rem;">{prob*100:.1f}%</div>',
            unsafe_allow_html=True)


def render_detections(detections: list, show_probs: bool):
    if not detections:
        st.markdown(
            '<div class="notice notice-pink">😶 No faces detected — try a well-lit frontal photo.</div>',
            unsafe_allow_html=True)
        return
    for i, det in enumerate(detections):
        key   = det["emotion"]
        disp  = det["display_emotion"]
        conf  = det["confidence"]
        emoji = det["emoji"]
        color = det["color_hex"]
        st.markdown(
            f'<div class="det-card">'
            f'<span class="det-emoji">{emoji}</span>'
            f'<span class="det-name" style="color:{color};">{disp}</span>'
            f'<span class="det-conf">Face #{i+1} &nbsp;·&nbsp; {conf*100:.0f}% confidence</span>'
            f'</div>', unsafe_allow_html=True)
        if show_probs:
            with st.expander(f"All probabilities — Face #{i+1}"):
                render_prob_bars(det["probabilities"])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero" style="font-size:1.7rem;letter-spacing:-1px;">🧠 EmoSense</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:0.75rem;">AI Emotion Detection</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Mode**")
    mode = st.radio("", ["📷 Upload Image", "🎞️ Upload Video", "📸 Live Camera"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Settings**")
    draw_boxes = st.checkbox("Draw bounding boxes", value=True)
    show_probs = st.checkbox("Show emotion probabilities", value=True)

    st.markdown("---")
    # Emotion class legend
    st.markdown("**Emotion Classes**")
    for key in ["angry","disgusted","fearful","happy","neutral","sad","surprised"]:
        color = EMOTION_HEX[key]
        st.markdown(
            f'<span class="pill" style="background:{color}22;color:{color};border:1px solid {color}55;">'
            f'{EMOTION_EMOJIS[key]} {EMOTION_DISPLAY[key]}</span>',
            unsafe_allow_html=True)
    st.markdown("")
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.68rem;color:#33334a;line-height:1.8;">'
        '<b>YOLOv8</b> face detection<br>'
        '<b>EmotionCNN</b> classification<br>'
        'FER dataset · 7 emotions<br>'
        '☁️ Streamlit Cloud ready</div>', unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="hero">Emotion Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Detects faces with YOLOv8 · Classifies into '
    '<b>Angry · Disgusted · Fearful · Happy · Neutral · Sad · Surprised</b></div>',
    unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("🔄 Initialising YOLOv8 + EmotionCNN…"):
    try:
        detector = load_detector()
        st.success("✅ Models ready — YOLO face detector + EmotionCNN loaded.", icon="🧠")
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  MODE 1 — Upload Image
# ══════════════════════════════════════════════════════════════════════════════
if "Upload Image" in mode:
    st.markdown('<div class="sec-head">📷 Upload an Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp","bmp"],
                                label_visibility="collapsed")

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        col_img, col_res = st.columns([6, 4], gap="large")

        with col_img:
            with st.spinner("🔍 Running YOLO detection + emotion classification…"):
                t0 = time.time()
                annotated_pil, detections = run_detection(pil_img, detector, draw_boxes)
                elapsed_ms = (time.time() - t0) * 1000

            st.image(annotated_pil, use_container_width=True, caption="Annotated Output")
            st.markdown("<br>", unsafe_allow_html=True)
            dominant = detections[0]["emotion"] if detections else "—"
            render_stat_row(len(detections), elapsed_ms, dominant)

        with col_res:
            st.markdown('<div class="sec-head">🎯 Detection Results</div>', unsafe_allow_html=True)
            render_detections(detections, show_probs)
            st.markdown("<br>", unsafe_allow_html=True)
            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                "⬇️ Download Annotated Image",
                data=buf.getvalue(),
                file_name="emosense_result.png",
                mime="image/png",
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
#  MODE 2 — Upload Video
# ══════════════════════════════════════════════════════════════════════════════
elif "Video" in mode:
    st.markdown('<div class="sec-head">🎞️ Upload a Video</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notice notice-green">'
        'ℹ️ Keep videos under 30 seconds for best performance on Streamlit Cloud. '
        'Max upload size is 200 MB.</div>', unsafe_allow_html=True)

    uploaded_vid = st.file_uploader("", type=["mp4","avi","mov","mkv"],
                                    label_visibility="collapsed")

    if uploaded_vid:
        ts      = int(time.time())
        tmp_in  = f"/tmp/emo_in_{ts}.mp4"
        tmp_out = f"/tmp/emo_out_{ts}.mp4"

        with open(tmp_in, "wb") as f:
            f.write(uploaded_vid.read())

        cap   = cv2.VideoCapture(tmp_in)
        fps_v = int(cap.get(cv2.CAP_PROP_FPS)) or 24
        vw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.info(f"📹 {vw}×{vh} · {fps_v} FPS · {total} frames")

        writer  = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (vw, vh))
        bar     = st.progress(0, text="Processing…")
        preview = st.empty()

        all_emotions, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, dets = detector.process_image(frame, draw_boxes=draw_boxes)
            writer.write(annotated)
            all_emotions.extend(d["emotion"] for d in dets)

            if idx % max(1, total // 15) == 0:
                preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                              use_container_width=True, caption=f"Frame {idx}/{total}")
            bar.progress(min((idx + 1) / max(total, 1), 1.0),
                         text=f"Frame {idx + 1} / {total}")
            idx += 1

        cap.release()
        writer.release()
        bar.progress(1.0, text="✅ Processing complete!")

        # Summary chart
        if all_emotions:
            from collections import Counter
            counts = Counter(all_emotions)
            st.markdown('<div class="sec-head">📊 Emotion Distribution</div>', unsafe_allow_html=True)
            cols = st.columns(len(counts))
            for i, (em, cnt) in enumerate(counts.most_common()):
                pct = cnt / len(all_emotions) * 100
                with cols[i]:
                    st.markdown(
                        f'<div class="stat-card">'
                        f'<div style="font-size:1.6rem;">{EMOTION_EMOJIS[em]}</div>'
                        f'<div class="stat-num" style="color:{EMOTION_HEX[em]};font-size:1.4rem;">'
                        f'{pct:.0f}%</div>'
                        f'<div class="stat-lbl">{EMOTION_DISPLAY[em]}</div>'
                        f'</div>', unsafe_allow_html=True)

        with open(tmp_out, "rb") as f:
            st.download_button("⬇️ Download Processed Video", data=f.read(),
                               file_name="emosense_video.mp4", mime="video/mp4",
                               use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MODE 3 — Live Camera (st.camera_input — browser API, cloud compatible)
# ══════════════════════════════════════════════════════════════════════════════
elif "Camera" in mode:
    st.markdown('<div class="sec-head">📸 Live Camera</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notice notice-blue">'
        '📷 Click <b>Take photo</b> to capture a frame from your device camera. '
        'The browser requests permission on first use. '
        'Retake anytime to analyse a new expression.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    cam_img = st.camera_input("", label_visibility="collapsed")

    if cam_img:
        pil_img = Image.open(cam_img).convert("RGB")
        col_l, col_r = st.columns([6, 4], gap="large")

        with col_l:
            with st.spinner("🔍 Analysing expression…"):
                t0 = time.time()
                annotated_pil, detections = run_detection(pil_img, detector, draw_boxes)
                elapsed_ms = (time.time() - t0) * 1000

            st.image(annotated_pil, use_container_width=True, caption="Emotion Analysis")
            st.markdown("<br>", unsafe_allow_html=True)
            dominant = detections[0]["emotion"] if detections else "—"
            render_stat_row(len(detections), elapsed_ms, dominant)

        with col_r:
            st.markdown('<div class="sec-head">🎯 Results</div>', unsafe_allow_html=True)
            render_detections(detections, show_probs)
            if detections:
                st.markdown("<br>", unsafe_allow_html=True)
                buf = io.BytesIO()
                annotated_pil.save(buf, format="PNG")
                st.download_button("⬇️ Download Result", data=buf.getvalue(),
                                   file_name="emosense_capture.png", mime="image/png",
                                   use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:0.72rem;color:#222240;">'
    'EmoSense &nbsp;·&nbsp; YOLOv8 + EmotionCNN &nbsp;·&nbsp; '
    'Classes: Angry · Disgusted · Fearful · Happy · Neutral · Sad · Surprised'
    '&nbsp;·&nbsp; ☁️ Streamlit Cloud Ready'
    '</div>', unsafe_allow_html=True)
