"""
main app/ ui/ ux
اجرا: streamlit run app/app.py
"""
import os
import sys
import json
import pickle

import joblib
import pandas as pd
import altair as alt
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import preprocess  # noqa: E402

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ------------------------------------------------------------------
# ظاهر
# ------------------------------------------------------------------
BG = "#12101F"
CARD_BG = "#1C1934"
CARD_BORDER = "#2C2750"
ACCENT = "#FF6B6B"
ACCENT_2 = "#6C63FF"
TEXT = "#F4F3F9"
MUTED = "#9B97B8"

EMOTIONS = {
    "ANGRY":    {"fa": "خشم",  "color": "#E4572E"},
    "FEAR":     {"fa": "ترس",  "color": "#7C6FDB"},
    "HAPPY":    {"fa": "شادی", "color": "#F2B134"},
    "HATE":     {"fa": "نفرت", "color": "#B23A48"},
    "SAD":      {"fa": "غم",   "color": "#4A7A96"},
    "SURPRISE": {"fa": "تعجب", "color": "#3FA796"},
    "OTHER":    {"fa": "سایر", "color": "#8A8A9E"},
}


def inject_css():
    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Vazirmatn', sans-serif !important;
    }}
    #MainMenu, header, footer {{ visibility: hidden; }}
    .stApp {{ background: {BG}; }}
    .main .block-container {{
        direction: rtl;
        max-width: 860px;
        padding-top: 2.2rem;
    }}

    .nyko-wordmark {{
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.15rem;
        direction: ltr;
    }}
    .nyko-tagline {{
        text-align: center;
        color: {MUTED};
        font-size: 0.95rem;
        margin-bottom: 2.2rem;
    }}

    .stTextArea textarea {{
        background: {CARD_BG} !important;
        border: 1px solid {CARD_BORDER} !important;
        border-radius: 14px !important;
        color: {TEXT} !important;
        font-family: 'Vazirmatn', sans-serif !important;
        font-size: 1rem !important;
    }}
    .stTextArea textarea:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 1px {ACCENT} !important;
    }}

    .stButton > button {{
        background: {ACCENT} !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
        transition: transform 0.15s ease, opacity 0.15s ease;
    }}
    .stButton > button:hover {{ transform: translateY(-1px); opacity: 0.92; }}
    .stButton > button p {{ font-family: 'Vazirmatn', sans-serif !important; font-weight: 700 !important; }}

    .result-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 16px;
        padding: 1.1rem 1.3rem 0.4rem 1.3rem;
        margin-bottom: 0.8rem;
    }}
    .result-model-name {{ color: {TEXT}; font-size: 0.95rem; font-weight: 700; }}
    .result-model-sub {{ color: {MUTED}; font-size: 0.78rem; margin-bottom: 0.7rem; }}
    .result-hero {{ display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.4rem; }}
    .result-hero-label {{ font-size: 1.5rem; font-weight: 800; }}
    .result-hero-score {{ font-size: 1.05rem; font-weight: 700; color: {MUTED}; }}

    div[data-testid="stPopover"] > div > button {{
        border-radius: 999px !important;
        border: 1px solid {CARD_BORDER} !important;
        background: {CARD_BG} !important;
        color: {MUTED} !important;
    }}
    div[data-testid="stPopoverBody"] {{ direction: rtl; text-align: right; }}
    </style>
    """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load(os.path.join(
        MODELS_DIR, "tfidf_vectorizer.joblib"))
    classic_model = joblib.load(os.path.join(
        MODELS_DIR, "best_classic_model.joblib"))

    deep_model = load_model(os.path.join(MODELS_DIR, "best_deep_model.keras"))
    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "deep_config.json"), "r", encoding="utf-8") as f:
        deep_config = json.load(f)

    return vectorizer, classic_model, deep_model, tokenizer, label_encoder, deep_config


def predict_classic(text, vectorizer, model):
    clean = preprocess(text, remove_stopwords=True)
    X = vectorizer.transform([clean])
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    return dict(zip(classes, proba))


def predict_deep(text, model, tokenizer, label_encoder, max_len):
    clean = preprocess(text, remove_stopwords=False)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=max_len,
                           padding="post", truncating="post")
    proba = model.predict(padded, verbose=0)[0]
    classes = label_encoder.classes_
    return dict(zip(classes, proba))


def render_header():
    col_info, col_mid, _ = st.columns([1, 6, 1])
    with col_info:
        with st.popover("ⓘ"):
            st.markdown(
                """
**علی نیکوخصال**
دانشجوی مهندسی کامپیوتر

زیر نظر: دکتر مینا مسعودی‌فر

---
دیتاست آموزش: **ArmanEmo**
                """
            )
    with col_mid:
        st.markdown('<div class="nyko-wordmark">Nyko</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="nyko-tagline">تحلیل احساس متن فارسی با هوش مصنوعی</div>',
            unsafe_allow_html=True,
        )


def render_result(proba_dict, model_name, model_sub):
    sorted_items = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
    top_key, top_score = sorted_items[0]
    top_meta = EMOTIONS.get(top_key, {"fa": top_key, "color": MUTED})

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-model-name">{model_name}</div>
            <div class="result-model-sub">{model_sub}</div>
            <div class="result-hero">
                <span class="result-hero-label" style="color:{top_meta['color']}">{top_meta['fa']}</span>
                <span class="result-hero-score">{top_score * 100:.1f}٪</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = pd.DataFrame(
        [
            {
                "key": k,
                "label": EMOTIONS.get(k, {"fa": k})["fa"],
                "prob": v,
                "color": EMOTIONS.get(k, {"color": MUTED})["color"],
            }
            for k, v in sorted_items
        ]
    )

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=5, size=13)
        .encode(
            y=alt.Y(
                "label:N",
                sort=df["label"].tolist(),
                title=None,
                axis=alt.Axis(labelColor=TEXT, labelFontSize=12,
                              domain=False, ticks=False),
            ),
            x=alt.X(
                "prob:Q",
                title=None,
                axis=None,
                scale=alt.Scale(
                    domain=[0, max(0.35, float(df["prob"].max()) * 1.1)]),
            ),
            color=alt.Color(
                "key:N",
                scale=alt.Scale(domain=list(
                    df["key"]), range=list(df["color"])),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="احساس"),
                alt.Tooltip("prob:Q", title="احتمال", format=".1%"),
            ],
        )
        .properties(height=205, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    st.altair_chart(chart, width="stretch")


def main():
    st.set_page_config(page_title="Nyko", page_icon="🎭", layout="centered")
    inject_css()
    render_header()

    vectorizer, classic_model, deep_model, tokenizer, label_encoder, deep_config = load_artifacts()

    text = st.text_area(
        "متن",
        height=130,
        placeholder="متن خود را به صورت فارسی وارد کنید. مثال : خیلی خوشحالم",
        label_visibility="collapsed",
    )

    clicked = st.button("تحلیل کن", type="primary", width="stretch")

    if clicked:
        if not text.strip():
            st.warning("لطفاً یک متن وارد کنید.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                classic_proba = predict_classic(
                    text, vectorizer, classic_model)
                render_result(classic_proba, "مدل آماری",
                              "TF-IDF + Logistic Regression")
            with col2:
                deep_proba = predict_deep(
                    text, deep_model, tokenizer, label_encoder, deep_config["max_len"])
                render_result(
                    deep_proba, "مدل عصبی", f"شبکه‌ی {deep_config['best_cell_type'].upper()} دوطرفه"
                )

            with st.expander("جزئیات فنی"):
                st.caption("متن پردازش‌شده برای مدل آماری:")
                st.code(preprocess(text, remove_stopwords=True), language=None)
                st.caption("متن پردازش‌شده برای مدل عصبی:")
                st.code(preprocess(text, remove_stopwords=False), language=None)


if __name__ == "__main__":
    main()
