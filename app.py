"""
app.py
------
A colorful dashboard for the cat vs dog classifier. Upload any photo and
get a verdict with a confidence breakdown.

Run with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import build_model  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
CLASSES = ["cat", "dog"]


def html_block(s: str) -> str:
    """Flatten a multi-line HTML string to a single line before passing to
    st.markdown(unsafe_allow_html=True). Streamlit's markdown renderer treats
    any line indented 4+ spaces as a code block, and Python source
    indentation easily produces that -- collapsing to one line sidesteps the
    issue entirely, including when interpolated values are empty strings
    that would otherwise leave stray blank lines."""
    return " ".join(line.strip() for line in s.splitlines() if line.strip())

# Vivid two-tone palette, tied directly to the two classes
CAT_A, CAT_B = "#FF7A45", "#FFB84C"   # coral -> gold gradient
DOG_A, DOG_B = "#3D5AFE", "#00C2D1"   # indigo -> cyan gradient
VIOLET = "#8B5CF6"
PINK = "#EC4899"
GREEN = "#10B981"
INK = "#1F2430"
INK_MUTED = "#6B7280"
SURFACE = "#FFFFFF"


# --------------------------------------------------------------------------
# Page setup + design system
# --------------------------------------------------------------------------
st.set_page_config(page_title="Cat or Dog?", page_icon="🐾", layout="centered")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700;9..144,800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {INK};
    }}

    .stApp {{
        background: radial-gradient(circle at 12% 8%, rgba(255,122,69,0.16), transparent 42%),
                    radial-gradient(circle at 88% 6%, rgba(61,90,254,0.16), transparent 42%),
                    radial-gradient(circle at 50% 100%, rgba(139,92,246,0.10), transparent 55%),
                    #FCFAF7;
    }}

    #MainMenu, header, footer {{ visibility: hidden; }}

    .block-container {{
        max-width: 780px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }}

    /* Hero banner */
    .hero-banner {{
        background: linear-gradient(120deg, {CAT_A} 0%, {PINK} 45%, {VIOLET} 75%, {DOG_A} 100%);
        border-radius: 26px;
        padding: 2.6rem 2.4rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 45px -20px rgba(139, 92, 246, 0.55);
        position: relative;
        overflow: hidden;
    }}
    .hero-banner::before {{
        content: "🐾";
        position: absolute;
        font-size: 9rem;
        opacity: 0.12;
        right: -1.2rem;
        top: -2rem;
        transform: rotate(-12deg);
    }}
    .eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.85);
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    .hero-title {{
        font-family: 'Fraunces', serif;
        font-weight: 800;
        font-size: 3rem;
        line-height: 1.05;
        margin: 0 0 0.6rem 0;
        color: white;
        text-shadow: 0 2px 18px rgba(0,0,0,0.12);
    }}
    .hero-sub {{
        font-size: 1.02rem;
        color: rgba(255,255,255,0.92);
        max-width: 48ch;
    }}

    div[data-testid="stFileUploaderDropzone"] {{
        background: {SURFACE};
        border: 2px dashed {VIOLET};
        border-radius: 16px;
    }}
    div[data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {PINK};
    }}

    .result-card {{
        background: {SURFACE};
        border-radius: 20px;
        padding: 1.9rem 2.1rem;
        margin-top: 1.6rem;
        box-shadow: 0 16px 40px -20px rgba(31, 36, 48, 0.28);
        border-top: 6px solid;
        border-image: linear-gradient(90deg, {CAT_A}, {PINK}, {DOG_A}) 1;
    }}

    .verdict-badge {{
        display: inline-block;
        font-family: 'Fraunces', serif;
        font-weight: 800;
        font-size: 2.3rem;
        text-transform: capitalize;
        padding: 0.1rem 0;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .verdict-cat {{ background-image: linear-gradient(90deg, {CAT_A}, {PINK}); }}
    .verdict-dog {{ background-image: linear-gradient(90deg, {DOG_A}, {GREEN}); }}

    .verdict-conf {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        color: {INK_MUTED};
        margin: 0.1rem 0 1.4rem 0;
        font-weight: 600;
    }}

    .uncertain-badge {{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #92400E;
        background: #FEF3C7;
        border: 1px solid #FDE68A;
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        margin-bottom: 0.9rem;
    }}

    /* Signature element: dual-tone gradient confidence meter */
    .meter-track {{
        display: flex;
        width: 100%;
        height: 16px;
        border-radius: 999px;
        overflow: hidden;
        background: #EEECE6;
        margin-bottom: 0.6rem;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.08);
    }}
    .meter-fill-cat {{
        background: linear-gradient(90deg, {CAT_A}, {CAT_B});
        height: 100%;
        transition: width 0.7s cubic-bezier(.22,1,.36,1);
    }}
    .meter-fill-dog {{
        background: linear-gradient(90deg, {DOG_A}, {DOG_B});
        height: 100%;
        transition: width 0.7s cubic-bezier(.22,1,.36,1);
    }}

    .meter-labels {{
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
    }}
    .meter-labels .cat-tag {{ color: {CAT_A}; }}
    .meter-labels .dog-tag {{ color: {DOG_A}; }}

    /* Colorful stat pills */
    .stats-row {{
        display: flex;
        gap: 0.9rem;
        margin-top: 2rem;
        flex-wrap: wrap;
    }}
    .pill {{
        flex: 1;
        min-width: 150px;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        color: white;
        box-shadow: 0 10px 22px -14px rgba(0,0,0,0.35);
    }}
    .pill-1 {{ background: linear-gradient(135deg, {VIOLET}, {PINK}); }}
    .pill-2 {{ background: linear-gradient(135deg, {GREEN}, {DOG_B}); }}
    .pill-3 {{ background: linear-gradient(135deg, {CAT_A}, {CAT_B}); }}
    .pill-value {{
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.35rem;
    }}
    .pill-label {{
        font-size: 0.78rem;
        opacity: 0.9;
        margin-top: 0.15rem;
    }}

    .footer-note {{
        text-align: center;
        color: {INK_MUTED};
        font-size: 0.82rem;
        margin-top: 3rem;
    }}

    div.stButton > button {{
        background: linear-gradient(90deg, {VIOLET}, {PINK});
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.55rem 1.5rem;
        font-weight: 700;
    }}
    div.stButton > button:hover {{
        background: linear-gradient(90deg, {PINK}, {CAT_A});
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    is_pretrained = checkpoint.get("is_pretrained", True)
    input_size = checkpoint.get("input_size", 224)
    val_acc = checkpoint.get("val_acc", None)

    model, _, _ = build_model(freeze_backbone=False, force_scratch=not is_pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, device, input_size, val_acc


def predict(model, device, input_size, image: Image.Image):
    tf = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
    return probs  # [p_cat, p_dog]


# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    html_block(
        """
        <div class="hero-banner">
            <div class="eyebrow">IMAGE CLASSIFIER</div>
            <div class="hero-title">Cat or Dog? 🐾</div>
            <div class="hero-sub">Upload any photo and the model will tell you which one
            it sees, with a full confidence breakdown.</div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

model_load_error = None
try:
    model, device, input_size, val_acc = load_model()
except Exception as e:
    model_load_error = str(e)

if model_load_error:
    st.error(
        "Couldn't load the trained model checkpoint at "
        f"`checkpoints/best_model.pt`.\n\nDetails: {model_load_error}"
    )
else:
    uploaded = st.file_uploader(
        "Drop a photo here, or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is not None:
        image = Image.open(uploaded)

        col1, col2 = st.columns([1, 1.3], gap="large")
        with col1:
            st.image(image, use_container_width=True)

        with col2:
            probs = predict(model, device, input_size, image)
            p_cat, p_dog = float(probs[0]), float(probs[1])
            pred_idx = 0 if p_cat >= p_dog else 1
            pred_label = CLASSES[pred_idx]
            pred_conf = probs[pred_idx]
            badge_class = "verdict-cat" if pred_idx == 0 else "verdict-dog"

            uncertain_html = ""
            if pred_conf < 0.65:
                uncertain_html = (
                    '<div class="uncertain-badge">⚠ LOW CONFIDENCE — could easily be either</div>'
                )

            st.markdown(
                html_block(
                    f"""
                    <div class="result-card">
                        {uncertain_html}
                        <div class="verdict-badge {badge_class}">{pred_label} 🐾</div>
                        <div class="verdict-conf">{pred_conf:.1%} confidence</div>
                        <div class="meter-track">
                            <div class="meter-fill-cat" style="width:{p_cat*100:.1f}%;"></div>
                            <div class="meter-fill-dog" style="width:{p_dog*100:.1f}%;"></div>
                        </div>
                        <div class="meter-labels">
                            <span class="cat-tag">🐱 CAT {p_cat:.1%}</span>
                            <span class="dog-tag">🐶 DOG {p_dog:.1%}</span>
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

    # ----------------------------------------------------------------
    # Colorful model stats
    # ----------------------------------------------------------------
    acc_display = f"{val_acc:.1%}" if val_acc else "—"
    st.markdown(
        html_block(
            f"""
            <div class="stats-row">
                <div class="pill pill-1">
                    <div class="pill-value">{acc_display}</div>
                    <div class="pill-label">Validation accuracy</div>
                </div>
                <div class="pill pill-2">
                    <div class="pill-value">25,000</div>
                    <div class="pill-label">Training images</div>
                </div>
                <div class="pill pill-3">
                    <div class="pill-value">{input_size}px</div>
                    <div class="pill-label">Input resolution</div>
                </div>
            </div>
            <div class="footer-note">Cat vs Dog Classifier &middot; PyTorch &middot; Built with Streamlit</div>
            """
        ),
        unsafe_allow_html=True,
    )