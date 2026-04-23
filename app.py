# =============================================================================
# NeuroScan AI — Brain Tumor Segmentation
# =============================================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import io
import time
from skimage.measure import label, regionprops

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMG_HEIGHT = 256
IMG_WIDTH  = 256
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "unet_brain_tumor.pth"


# ─────────────────────────────────────────────
# CUSTOM CSS — Refined Clinical Dark Theme
# ─────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0a0c14;
    --surface:   #111420;
    --card:      #181d2e;
    --border:    rgba(99, 120, 255, 0.12);
    --accent:    #6378ff;
    --accent2:   #00d4aa;
    --danger:    #ff4d6d;
    --warn:      #fbbf24;
    --text:      #dde3f0;
    --muted:     #6b7594;
    --label:     #a0aabe;
}

*, *::before, *::after { box-sizing: border-box; }

/* ── Hide Streamlit's default white top bar ── */
header[data-testid="stHeader"],
.stAppHeader,
#stDecoration {
    display: none !important;
    height: 0 !important;
}

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stCaption {
    color: var(--label) !important;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}
p, span, label, .stMarkdown, .stCaption {
    color: var(--label) !important;
}
code, .stCode {
    font-family: 'DM Mono', monospace !important;
    color: var(--accent2) !important;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(99, 120, 255, 0.35) !important;
    border-radius: 12px !important;
    background: var(--card) !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: var(--label) !important;
}
[data-testid="stFileUploader"] small {
    color: var(--muted) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.03em;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    transition: opacity 0.2s !important;
}
.stButton > button p,
.stButton > button span {
    color: #ffffff !important;
    font-weight: 700 !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}
.stDownloadButton > button {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--accent2) !important;
    font-size: 13px !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent2) !important;
}

/* ── Metrics ── */
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: 8px;
    padding: 3px;
    gap: 2px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    padding: 7px 18px !important;
    color: var(--muted) !important;
    font-size: 13px !important;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 120, 255, 0.15) !important;
    color: var(--accent) !important;
}

/* ── Progress ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 4px;
}
.stProgress > div > div {
    background: var(--card) !important;
    border-radius: 4px;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Alerts ── */
.stAlert {
    border-radius: 8px !important;
    background: rgba(251, 191, 36, 0.08) !important;
    border: 1px solid rgba(251, 191, 36, 0.2) !important;
    color: var(--warn) !important;
}
.stAlert p, .stAlert span {
    color: var(--warn) !important;
}

/* ── Checkbox ── */
.stCheckbox label span {
    color: var(--label) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Info box ── */
.stInfo {
    background: rgba(99, 120, 255, 0.07) !important;
    border: 1px solid rgba(99, 120, 255, 0.2) !important;
    border-radius: 8px !important;
}
.stInfo p, .stInfo span { color: var(--label) !important; }

/* ── Selectbox ── */
.stSelectbox label { color: var(--label) !important; }
</style>
"""


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE — U-Net
# ─────────────────────────────────────────────
class DoubleConv(nn.Module):
    """Conv→BN→ReLU→Conv→BN→ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net"""
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.ups   = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        features = [64, 128, 256, 512]
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model weights not found. Place 'unet_brain_tumor.pth' next to app.py.")
        return None
    except RuntimeError as e:
        st.error(f"Weight loading error: {e}")
        return None


# ─────────────────────────────────────────────
# PREPROCESSING — Multi-channel CLAHE
# ─────────────────────────────────────────────
def preprocess_image(image_pil):
    img = np.array(image_pil)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # CLAHE in LAB space — improves local contrast, safe for model inputs
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    img_enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)

    # NOTE: No unsharp masking — shifts pixel distribution from training data
    # and causes near-zero sigmoid outputs from the model.
    return img, img_enhanced


# ─────────────────────────────────────────────
# INFERENCE — Single Pass
# ─────────────────────────────────────────────
_BASE_TRANSFORM = A.Compose([
    A.Resize(IMG_HEIGHT, IMG_WIDTH),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2(),
])


def _infer(img_np, model, transform=None):
    tf = transform or _BASE_TRANSFORM
    t  = tf(image=np.ascontiguousarray(img_np))["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(t)).cpu().numpy()[0, 0]


# ─────────────────────────────────────────────
# TEST-TIME AUGMENTATION — 4-fold (safe augmentations only)
# ─────────────────────────────────────────────
def predict_tta(img_enhanced, model):
    """
    4 safe augmentations for brain MRI:
      - Original
      - Horizontal flip  (brain is left-right symmetric → valid)
      - Slight brightness up
      - Slight brightness down

    V-flip and 90°/270° rotations are intentionally excluded — brain MRI
    has a fixed anatomical orientation, so those produce out-of-distribution
    images that dilute (and therefore suppress) tumor probabilities.
    """
    preds = []

    def infer(im):
        return _infer(np.ascontiguousarray(im), model)

    img = img_enhanced

    # 1. Original
    preds.append(infer(img))

    # 2. Horizontal flip (left-right brain symmetry is valid)
    p_hflip = infer(np.fliplr(img))
    preds.append(np.fliplr(p_hflip))

    # 3. Slight brightness boost (+15)
    bright_up = np.clip(img.astype(np.int16) + 15, 0, 255).astype(np.uint8)
    preds.append(infer(bright_up))

    # 4. Slight brightness reduction (-15)
    bright_dn = np.clip(img.astype(np.int16) - 15, 0, 255).astype(np.uint8)
    preds.append(infer(bright_dn))

    return np.mean(preds, axis=0)


# ─────────────────────────────────────────────
# ADAPTIVE THRESHOLDING
# ─────────────────────────────────────────────
def adaptive_threshold(prob_map):
    """
    Two-stage threshold selection:
    1. Otsu's method on the probability map
    2. Fallback to top-5% percentile if Otsu yields no detections

    Hard floor is 0.10 (not 0.3) so low-confidence models still detect.
    Hard ceiling is 0.70 to prevent over-segmentation.
    """
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(
        prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu = otsu_val / 255.0

    # If Otsu threshold exceeds 95th percentile → nothing would be detected;
    # fall back to the 95th percentile so the brightest 5% is always kept.
    p95 = float(np.percentile(prob_map, 95))
    if otsu > p95:
        otsu = p95 * 0.85          # slightly below p95 to capture the region

    return float(np.clip(otsu, 0.10, 0.70))


# ─────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────
def postprocess_mask(mask_u8, min_area_ratio=0.003):
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k_close, iterations=2)
    m = cv2.morphologyEx(m,       cv2.MORPH_OPEN,  k_open,  iterations=1)

    min_px  = int(m.shape[0] * m.shape[1] * min_area_ratio)
    labeled = label(m > 0)
    cleaned = np.zeros_like(m)
    for r in regionprops(labeled):
        if r.area >= min_px:
            for c in r.coords:
                cleaned[c[0], c[1]] = 255
    return cleaned


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
def create_overlay(img_rgb, mask, alpha=0.40):
    colored = np.zeros_like(img_rgb)
    colored[mask > 0] = [80, 120, 255]   # cool blue highlight
    return cv2.addWeighted(img_rgb, 1 - alpha, colored, alpha, 0)


def create_heatmap(prob_map):
    return cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def compute_metrics(mask, prob_map):
    total = mask.shape[0] * mask.shape[1]
    tpx   = np.count_nonzero(mask)
    pct   = tpx / total * 100
    conf  = float(np.mean(prob_map[mask > 0]) * 100) if tpx > 0 else 0.0

    if   pct < 0.5: sev, col = "None",     "#00d4aa"
    elif pct < 1.5: sev, col = "Low",      "#6378ff"
    elif pct < 4:   sev, col = "Moderate", "#fbbf24"
    elif pct < 8:   sev, col = "High",     "#f97316"
    else:           sev, col = "Critical", "#ff4d6d"

    return {"pct": round(pct, 2), "conf": round(conf, 1),
            "severity": sev, "sev_color": col, "tpx": tpx}


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
def predict(image_pil, model, use_tta=True):
    t0 = time.time()
    img_np, img_enh = preprocess_image(image_pil)

    prob = predict_tta(img_enh, model) if use_tta else _infer(img_enh, model)
    thresh    = adaptive_threshold(prob)
    mask_raw  = (prob > thresh).astype(np.uint8) * 255
    mask      = postprocess_mask(mask_raw)

    img_r = cv2.resize(img_np, (IMG_WIDTH, IMG_HEIGHT))
    overlay = create_overlay(img_r, mask)
    heatmap = create_heatmap(prob)
    metrics = compute_metrics(mask, prob)
    metrics["ms"]     = round((time.time() - t0) * 1000)
    metrics["thresh"] = round(thresh, 3)

    return {"original": img_r, "mask": mask, "overlay": overlay,
            "heatmap": heatmap, "prob": prob, "metrics": metrics}


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="NeuroScan AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<h2 style='font-size:1.1rem; letter-spacing:-0.01em;"
            " margin-bottom:4px;'>NeuroScan AI</h2>"
            "<p style='font-size:12px; margin-top:0;'>U-Net + TTA</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Analysis settings
        st.markdown(
            "<p style='font-size:11px; text-transform:uppercase; "
            "letter-spacing:0.08em; color:#6b7594;'>Analysis</p>",
            unsafe_allow_html=True,
        )
        use_tta = st.checkbox("Test-Time Augmentation (4×)", value=True,
                              help="Ensemble of 4 passes (original + h-flip + brightness variants) for more robust predictions.")
        overlay_alpha = st.slider("Overlay opacity", 0.2, 0.7, 0.4, 0.05)

        st.divider()

        # Model info
        st.markdown(
            "<p style='font-size:11px; text-transform:uppercase; "
            "letter-spacing:0.08em; color:#6b7594;'>Model</p>",
            unsafe_allow_html=True,
        )
        device_badge = (
            "<span style='background:#00d4aa22; color:#00d4aa; padding:2px 8px;"
            " border-radius:4px; font-size:11px; font-family:DM Mono,monospace;'>"
            f"{DEVICE.upper()}</span>"
        )
        st.markdown(
            f"Architecture &nbsp;&nbsp; **U-Net**<br>"
            f"Input size &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **256 × 256**<br>"
            f"Device &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {device_badge}",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            "<p style='font-size:12px; color:#6b7594; line-height:1.6;'>"
            "Upload an axial Brain MRI slice (TCGA-LGG or similar).<br><br>"
            "Supports: JPG · PNG · TIF · BMP</p>",
            unsafe_allow_html=True,
        )

    # ── Main ─────────────────────────────────────────────────────────────────
    # Header
    st.markdown(
        "<div style='padding:1.5rem 0 0.5rem;'>"
        "<h1 style='font-size:2rem; margin:0;'>Brain Tumor Segmentation</h1>"
        "<p style='margin:6px 0 0; font-size:14px;'>"
        "U-Net · CLAHE enhancement · 4× TTA · Adaptive thresholding</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    model = load_model()
    if model is None:
        st.stop()

    # Upload
    uploaded = st.file_uploader(
        "Upload Brain MRI Scan",
        type=["jpg", "png", "tif", "jpeg", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(
            "<div style='text-align:center; padding:3rem 0; color:#6b7594; font-size:14px;'>"
            "Drop a brain MRI image above to begin analysis.</div>",
            unsafe_allow_html=True,
        )
        return

    image = Image.open(uploaded)

    # Layout: preview + button side by side
    col_img, col_btn = st.columns([1, 2])
    with col_img:
        st.markdown(
            "<p style='font-size:11px; text-transform:uppercase; letter-spacing:0.08em;"
            " color:#6b7594; margin-bottom:6px;'>Uploaded scan</p>",
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)
        st.markdown(
            f"<p style='font-size:11px; color:#6b7594; margin-top:4px;'>"
            f"{uploaded.name} · {image.size[0]}×{image.size[1]} px</p>",
            unsafe_allow_html=True,
        )

    with col_btn:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        run = st.button("Run Segmentation", use_container_width=False)
        if run:
            with st.spinner("Analysing…"):
                res = predict(image, model, use_tta=use_tta)
                # re-create overlay with sidebar alpha
                img_r = res["original"]
                mask  = res["mask"]
                colored = np.zeros_like(img_r)
                colored[mask > 0] = [80, 120, 255]
                res["overlay"] = cv2.addWeighted(
                    img_r, 1 - overlay_alpha, colored, overlay_alpha, 0
                )
                st.session_state["res"] = res

    # ── Results ──────────────────────────────────────────────────────────────
    if "res" not in st.session_state:
        return

    res = st.session_state["res"]
    m   = res["metrics"]

    st.divider()

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tumor Area",  f"{m['pct']}%")
    c2.metric("Confidence",  f"{m['conf']}%")
    c3.metric("Severity",    m["severity"])
    c4.metric("Process time",f"{m['ms']} ms")
    c5.metric("Threshold",   f"{m['thresh']}")

    # Confidence bar
    st.progress(min(m["conf"] / 100.0, 1.0))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Tabs with compact images
    tab1, tab2, tab3, tab4 = st.tabs(["Overlay", "Side-by-Side", "Heatmap", "Mask"])

    IMG_COLS = 2   # use 2-column layout inside tabs to keep images compact

    with tab1:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.image(res["overlay"], caption="Blue region = detected tumor",
                     use_container_width=True)

    with tab2:
        ca, cb = st.columns(2)
        with ca:
            st.markdown(
                "<p style='font-size:11px; text-transform:uppercase;"
                " letter-spacing:0.07em; color:#6b7594;'>Original</p>",
                unsafe_allow_html=True,
            )
            st.image(res["original"], use_container_width=True)
        with cb:
            st.markdown(
                "<p style='font-size:11px; text-transform:uppercase;"
                " letter-spacing:0.07em; color:#6b7594;'>Segmentation</p>",
                unsafe_allow_html=True,
            )
            st.image(res["overlay"], use_container_width=True)

    with tab3:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            hm_rgb = cv2.cvtColor(res["heatmap"], cv2.COLOR_BGR2RGB)
            st.image(hm_rgb, caption="Confidence map — dark→low · bright→high",
                     use_container_width=True)

    with tab4:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.image(res["mask"], caption="Binary segmentation mask",
                     use_container_width=True)

    # Downloads
    st.divider()
    st.markdown(
        "<p style='font-size:11px; text-transform:uppercase; letter-spacing:0.08em;"
        " color:#6b7594;'>Export</p>",
        unsafe_allow_html=True,
    )
    d1, d2, d3, d4 = st.columns(4)

    def img_bytes(arr, bgr=False):
        if bgr:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return buf.getvalue()

    d1.download_button("⬇ Mask",    img_bytes(res["mask"]),    "tumor_mask.png",    "image/png", use_container_width=True)
    d2.download_button("⬇ Overlay", img_bytes(res["overlay"]), "tumor_overlay.png", "image/png", use_container_width=True)
    d3.download_button("⬇ Heatmap", img_bytes(res["heatmap"], bgr=True), "heatmap.png", "image/png", use_container_width=True)

    prob_img = (res["prob"] * 255).astype(np.uint8)
    d4.download_button("⬇ Prob map", img_bytes(prob_img), "prob_map.png", "image/png", use_container_width=True)

    # Disclaimer
    st.warning(
        "⚠️ **Research use only.** This tool is not a medical device. "
        "Always seek a qualified radiologist for clinical diagnosis."
    )


if __name__ == "__main__":
    main()