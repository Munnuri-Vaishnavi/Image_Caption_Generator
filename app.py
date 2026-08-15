"""
Image Caption Generator - Streamlit Demo
Week 8 - Deployment prep

Loads the ResNet50 + LSTM model trained in the Colab notebook (Weeks 4-7)
and lets a user upload an image (or pick a sample) to generate a caption,
with optional translation and audio playback.

Run locally with:
    streamlit run app.py

Expected files in the same folder as this script:
    ResNet50_LSTM_Flickr8k_caption_model.keras
    Flickr8k_word_index.json
    caption_model_config.json
    samples/               (a folder of pre-picked images - optional but recommended)

Additional packages needed beyond the core ones:
    pip install gTTS deep-translator
"""

import os
import io
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from huggingface_hub import hf_hub_download

# ---------- Hugging Face Hub location of the model ----------
HF_REPO_ID = "vaishnavi22092006/Caption_.Generator"
HF_MODEL_FILENAME = "ResNet50_LSTM_Flickr8k_caption_model.keras"

WORD_INDEX_PATH = "Flickr8k_word_index.json"
CONFIG_PATH = "caption_model_config.json"
SAMPLES_DIR = "samples"

LANGUAGE_OPTIONS = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    "Kannada": "kn",
}

st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️", layout="wide")

# ---------------------------------------------------------------------------
# Theme toggle (manual CSS-based light/dark, since Streamlit can't switch
# .streamlit/config.toml themes at runtime without a restart)
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

THEMES = {
    "dark": {
        "bg": "#0e1117", "card": "#1c1f26", "text": "#f5f5f5",
        "subtext": "#9ca3af", "accent": "#6366f1",
    },
    "light": {
        "bg": "#ffffff", "card": "#f3f4f6", "text": "#111827",
        "subtext": "#6b7280", "accent": "#4f46e5",
    },
}
t = THEMES[st.session_state.theme]

st.markdown(f"""
<style>
    /* Main app area */
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {t['bg']};
        color: {t['text']};
    }}
    [data-testid="stHeader"] {{
        background-color: {t['bg']};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {t['card']};
    }}
    [data-testid="stSidebar"] * {{
        color: {t['text']} !important;
    }}

    /* Body text, headings, captions */
    p, span, label, h1, h2, h3, h4, .stMarkdown, .stCaption {{
        color: {t['text']} !important;
    }}
    .app-subtext, [data-testid="stCaptionContainer"] {{
        color: {t['subtext']} !important;
    }}

    /* All Streamlit buttons share this underlying testid, regardless of which
       wrapper widget renders them (st.button, st.download_button, and the
       file_uploader's internal "Browse files" button all use it) */
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-minimal"],
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-minimal"],
    button[data-testid="stBaseButton-secondaryFormSubmit"],
    button[kind="secondary"],
    button[kind="primary"],
    button[kind="minimal"],
    [data-testid="stButton"] button,
    [data-testid="stDownloadButton"] button,
    [data-testid="stFormSubmitButton"] button {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
        border: 1px solid {t['accent']} !important;
    }}
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-minimal"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="stBaseButton-minimal"]:hover,
    button[kind="secondary"]:hover,
    button[kind="primary"]:hover,
    button[kind="minimal"]:hover,
    [data-testid="stButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover {{
        background-color: {t['accent']} !important;
        color: #ffffff !important;
        border: 1px solid {t['accent']} !important;
    }}

    /* Disabled buttons (e.g. Download button before results exist) get their own
       default styling in Streamlit that ignores the rules above - cover explicitly,
       including aria-disabled since some Streamlit versions use that instead of
       the plain HTML disabled attribute */
    button:disabled, button[disabled], button[aria-disabled="true"],
    [data-testid="stButton"] button:disabled,
    [data-testid="stDownloadButton"] button:disabled,
    [data-testid="stButton"] button[aria-disabled="true"],
    [data-testid="stDownloadButton"] button[aria-disabled="true"] {{
        background-color: {t['card']} !important;
        color: {t['subtext']} !important;
        border: 1px solid {t['subtext']} !important;
        opacity: 0.6;
    }}

    /* Icons (e.g. download/speaker icons) inside buttons should inherit the theme
       text color instead of defaulting to black/dark, which is invisible on dark bg */
    button svg {{
        fill: {t['text']} !important;
    }}
    button:disabled svg, button[disabled] svg {{
        fill: {t['subtext']} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {t['card']};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {t['text']};
    }}
    .stTabs [aria-selected="true"] {{
        color: {t['accent']} !important;
    }}

    /* Selectbox / dropdown - wildcard override on every descendant, since Streamlit's
       BaseWeb select has more nested layers than the specific selectors above catch */
    [data-testid="stSelectbox"] * {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
    }}
    [data-baseweb="popover"] * {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
    }}
    li[data-baseweb="menu-item"]:hover {{
        background-color: {t['accent']} !important;
    }}

    /* Dropdown option list, targeted via ARIA roles instead of data-baseweb attributes -
       roles are far more stable across BaseWeb/Streamlit version changes than internal
       data-* attribute names, which have already been renamed once in this project */
    [role="listbox"] {{
        background-color: {t['card']} !important;
    }}
    [role="option"] {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
    }}
    [role="option"]:hover, [role="option"][aria-selected="true"] {{
        background-color: {t['accent']} !important;
        color: #ffffff !important;
    }}

    [data-testid="stSelectbox"] label {{
        color: {t['text']} !important;
    }}

    /* Expander - wildcard + !important, since the clickable header/summary bar
       has its own default white background that the plain rule above didn't override */
    [data-testid="stExpander"] {{
        background-color: {t['card']} !important;
        border-radius: 8px;
    }}
    [data-testid="stExpander"] * {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
    }}
    [data-testid="stExpander"] summary {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
    }}
    details {{
        background-color: {t['card']} !important;
    }}
    summary {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
    }}

    /* Info / success / warning boxes */
    [data-testid="stAlert"] {{
        background-color: {t['card']};
        color: {t['text']};
    }}

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {t['card']} !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{
        color: {t['text']} !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] {{
        color: {t['text']} !important;
    }}

    /* Uploaded file preview chip (filename, size, remove button) shown after
       a file is selected - separate element from the dropzone above */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileData"] {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
    }}
    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileData"] * {{
        background-color: {t['card']} !important;
        color: {t['text']} !important;
    }}

    .app-card {{
        background-color: {t['card']};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.6rem;
    }}

    /* Hide Streamlit's built-in image fullscreen/maximize icon.
       Several selectors, since the exact markup/attribute Streamlit uses for
       this button has changed across versions - covering all known variants. */
    button[title="View fullscreen"],
    button[aria-label="View fullscreen"],
    [data-testid="StyledFullScreenButton"],
    [data-testid="stImageFullScreen"],
    [data-testid="stElementToolbar"],
    [data-testid="stFullScreenFrame"] button {{
        display: none !important;
        visibility: hidden !important;
    }}
</style>
""", unsafe_allow_html=True)

theme_col1, theme_col2 = st.columns([6, 1])
with theme_col2:
    toggle_label = "🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()


# ---------- Compatibility patch ----------
# Models saved with a newer Keras version can include config options (e.g.
# 'quantization_config') that an older local Keras install doesn't recognize yet,
# causing a deserialization error for each affected layer type in turn. This patch
# makes every layer type used in this model ignore unknown keyword arguments instead
# of crashing. The most robust real fix, if available to you, is:
#   pip install --upgrade tensorflow

_UNKNOWN_KWARGS_TO_STRIP = ["quantization_config"]
_LAYER_CLASSES_TO_PATCH = [
    tf.keras.layers.Dense,
    tf.keras.layers.Embedding,
    tf.keras.layers.LSTM,
    tf.keras.layers.Dropout,
    tf.keras.layers.Add,
]


def _make_patched_init(original_init):
    def _patched_init(self, *args, **kwargs):
        for key in _UNKNOWN_KWARGS_TO_STRIP:
            kwargs.pop(key, None)
        original_init(self, *args, **kwargs)
    return _patched_init


for _cls in _LAYER_CLASSES_TO_PATCH:
    if hasattr(_cls, "__init__"):
        _cls.__init__ = _make_patched_init(_cls.__init__)


# ---------- Loading (cached so this only happens once per session) ----------

@st.cache_resource(show_spinner=False)
def load_everything():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    with open(WORD_INDEX_PATH) as f:
        word_index = json.load(f)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def masked_loss(y_true, y_pred):
        loss = loss_object(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), loss.dtype)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    # Downloads the .keras file from the Hugging Face Hub repo the first time the
    # app runs, then caches it on disk - so it's only fetched once per deployment,
    # not on every user interaction (load_everything itself is also st.cache_resource).
    model_local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        # token=st.secrets["HF_TOKEN"],  # uncomment if your HF repo is private
    )

    caption_model = tf.keras.models.load_model(
        model_local_path, custom_objects={"masked_loss": masked_loss}
    )

    resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    resnet.trainable = False

    id_to_word = {idx: word for word, idx in word_index.items() if idx < config["vocab_size"]}

    return caption_model, resnet, word_index, id_to_word, config


# ---------- Core inference (mirrors the training notebook exactly) ----------

def image_to_feature(resnet, pil_image):
    img = pil_image.convert("RGB").resize((224, 224))
    arr = keras_image.img_to_array(img)
    arr = preprocess_input(np.expand_dims(arr, axis=0))
    return resnet.predict(arr, verbose=0)[0].astype("float32")


def beam_search_captions_multi(caption_model, id_to_word, config, feature, beam_width=3,
                                length_penalty=0.7, num_results=3):
    """Returns up to num_results (caption_text, normalized_score) tuples, best first."""
    max_length = config["max_length"]
    start_id = config["start_id"]
    end_id = config["end_id"]

    beams = [([start_id], 0.0)]
    completed = []

    for _ in range(max_length - 1):
        candidates = []
        for sequence, score in beams:
            if sequence[-1] == end_id:
                completed.append((sequence, score))
                continue

            input_sequence = np.zeros((1, max_length - 1), dtype="int32")
            usable = min(len(sequence), max_length - 1)
            input_sequence[0, :usable] = sequence[:usable]

            predictions = caption_model.predict([feature[None, :], input_sequence], verbose=0)[0]
            position = min(len(sequence) - 1, predictions.shape[0] - 1)
            probabilities = tf.nn.softmax(predictions[position]).numpy()
            probabilities[0] = 0.0

            best_ids = np.argsort(probabilities)[-beam_width:][::-1]
            for word_id in best_ids:
                probability = max(float(probabilities[word_id]), 1e-10)
                new_sequence = sequence + [int(word_id)]
                new_score = score + np.log(probability)
                candidates.append((new_sequence, new_score))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[1] / (len(item[0]) ** length_penalty), reverse=True)
        beams = candidates[:beam_width]

        if all(seq[-1] == end_id for seq, _ in beams):
            completed.extend(beams)
            break

    if not completed:
        completed = beams

    def seq_to_text(seq):
        words = []
        for word_id in seq:
            if word_id in (start_id, 0):
                continue
            if word_id == end_id:
                break
            words.append(id_to_word.get(word_id, "<unk>"))
        return " ".join(words)

    seen = {}
    for seq, score in completed:
        norm_score = score / (len(seq) ** length_penalty)
        text = seq_to_text(seq)
        if text and (text not in seen or norm_score > seen[text]):
            seen[text] = norm_score

    ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    return ranked[:num_results] if ranked else [("(no caption generated)", 0.0)]


def score_to_confidence_pct(score):
    """Rough, human-friendly confidence readout from a normalized log-probability score.
    Not a calibrated probability - just a visual indicator of relative certainty."""
    return max(0, min(100, int((score + 3.0) / 3.0 * 100)))


def generate_captions(pil_image):
    caption_model, resnet, word_index, id_to_word, config = load_everything()
    feature = image_to_feature(resnet, pil_image)
    return beam_search_captions_multi(caption_model, id_to_word, config, feature, num_results=3)


# ---------- Translation + audio (need internet - fail gracefully if unavailable) ----------

def translate_text(text, target_lang_code):
    if target_lang_code == "en":
        return text
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=target_lang_code).translate(text)
    except Exception as e:
        st.warning(f"Translation unavailable right now ({e}). Showing English instead.")
        return text


def text_to_speech_bytes(text, lang_code):
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=lang_code).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        st.warning(f"Audio generation unavailable right now ({e}).")
        return None


# ---------- Downloadable captioned image ----------

def create_captioned_image(pil_image, caption_text):
    img = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(16, W // 30))
    except Exception:
        font = ImageFont.load_default()

    words = caption_text.split()
    lines, current = [], ""
    for w in words:
        test = (current + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > W - 20 and current:
            lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)

    line_height = draw.textbbox((0, 0), "Ay", font=font)[3] + 6
    banner_height = line_height * len(lines) + 20
    draw.rectangle([0, H - banner_height, W, H], fill=(0, 0, 0, 160))

    y = H - banner_height + 10
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (W - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- Result display ----------
# FIX: this now reads from st.session_state instead of only running inside an
# `if st.button("Generate caption")` block. That old structure meant any nested
# button click (e.g. "Generate audio") triggered a full script rerun in which the
# outer button's clicked-state reset to False, so this whole block - including the
# audio logic - silently never ran again. Storing results in session_state and
# rendering them unconditionally on every rerun fixes that for audio, translation,
# and download all at once.

def show_results(state_key, pil_image):
    result = st.session_state.get(state_key)
    if result is None:
        return

    best_caption, results = result["best_caption"], result["results"]

    st.success(f"**Caption:** {best_caption}")

    with st.expander("See alternative captions the model considered"):
        for i, (text, score) in enumerate(results):
            pct = score_to_confidence_pct(score)
            st.write(f"{i + 1}. {text}")
            st.progress(pct / 100, text=f"~{pct}% relative confidence")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**🌐 Translate & listen**")
        lang_name = st.selectbox("Language", list(LANGUAGE_OPTIONS.keys()), key=f"lang_{state_key}")
        lang_code = LANGUAGE_OPTIONS[lang_name]
        translated = translate_text(best_caption, lang_code)
        if lang_code != "en":
            st.write(f"*{translated}*")
        if st.button("🔊 Generate audio", key=f"audio_btn_{state_key}"):
            with st.spinner("Generating audio..."):
                audio_bytes = text_to_speech_bytes(translated, lang_code)
            if audio_bytes:
                st.session_state[f"audio_{state_key}"] = audio_bytes

        if st.session_state.get(f"audio_{state_key}"):
            st.audio(st.session_state[f"audio_{state_key}"], format="audio/mp3")

    with col_b:
        st.markdown("**📥 Download**")
        captioned_bytes = create_captioned_image(pil_image, best_caption)
        st.download_button(
            "Download captioned image",
            data=captioned_bytes,
            file_name="captioned_image.png",
            mime="image/png",
            key=f"download_btn_{state_key}",
        )


def run_caption_generation(state_key, pil_image):
    """Computes captions once and stores them in session_state, and records history once."""
    with st.spinner("Generating caption..."):
        results = generate_captions(pil_image)
    best_caption, _ = results[0]

    st.session_state[state_key] = {"best_caption": best_caption, "results": results}
    st.session_state.pop(f"audio_{state_key}", None)  # clear any stale audio from a previous image

    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, best_caption)
    st.session_state.history = st.session_state.history[:10]

    st.rerun()  # forces the sidebar (rendered earlier in script order) to reflect the new history immediately


# ---------------------------- UI ----------------------------

st.title("🖼️ Image Caption Generator")
st.caption("ResNet50 (CNN) + LSTM (RNN), trained on Flickr8k — EDP AI/ML Project, Weeks 4-8")

st.info(
    "This model was trained on Flickr8k, which mostly contains everyday photos of "
    "people, animals, and outdoor activities. It performs best on similar images — "
    "results on very different subjects (objects, food, buildings) may be less accurate.",
    icon="ℹ️",
)

with st.sidebar:
    st.header("Recent captions")
    if st.session_state.get("history"):
        for c in st.session_state.history:
            st.write(f"• {c}")
    else:
        st.write("Generate a caption to see it appear here.")

tab1, tab2 = st.tabs(["📤 Upload your own image", "🖼️ Try a sample image"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_col, result_col = st.columns([1, 1.3])
        pil_image = Image.open(uploaded_file)
        with img_col:
            st.image(pil_image, use_container_width=True)
        with result_col:
            if st.button("Generate caption", type="primary", key="upload_btn"):
                run_caption_generation("upload_result", pil_image)
            show_results("upload_result", pil_image)

with tab2:
    if os.path.isdir(SAMPLES_DIR) and len(os.listdir(SAMPLES_DIR)) > 0:
        sample_files = sorted(
            f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        # Simple sequential labels: Image 1, Image 2, ... regardless of actual filename.
        # This does NOT rename files on disk - see rename_samples.py for that.
        display_labels = [f"Image {i + 1}" for i in range(len(sample_files))]
        label_to_file = dict(zip(display_labels, sample_files))
        selected_label = st.selectbox(
            f"Choose one of {len(sample_files)} sample images",
            display_labels,
        )

        if selected_label:
            selected_sample = label_to_file[selected_label]
            img_col, result_col = st.columns([1, 1.3])
            pil_image = Image.open(os.path.join(SAMPLES_DIR, selected_sample))
            with img_col:
                st.image(pil_image, use_container_width=True)
            with result_col:
                if st.button("Generate caption", type="primary", key="sample_btn"):
                    run_caption_generation("sample_result", pil_image)
                show_results("sample_result", pil_image)
    else:
        st.write(
            "No sample images found. Add a few images to a `samples/` folder next to this "
            "script to enable this tab."
        )

st.divider()
with st.expander("About this model"):
    st.markdown(
        """
        - **Architecture:** ResNet50 (pretrained CNN, frozen) extracts a 2048-number feature
          summary of the image; an LSTM (RNN) generates the caption one word at a time,
          conditioned on that summary.
        - **Training data:** Flickr8k (~8,091 images, 5 captions each).
        - **Generation method:** Beam search (width 3) with length normalization; top 3
          candidate captions are shown so you can see the model's alternatives.
        - **Known limitation:** as a small-dataset, non-attention model, this architecture
          compresses the whole image into a single fixed-size vector before generating any
          text, so fine visual detail can be lost — even on images similar to training data.
          This is the well-documented reason attention mechanisms and Vision Transformers
          were later developed for this task.
        """
    )
