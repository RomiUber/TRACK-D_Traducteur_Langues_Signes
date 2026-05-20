import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CLASSES    = ['A', 'B', 'C', 'L', 'W', 'Y']
MODEL_PATH = './model_weights'
if not os.path.exists(MODEL_PATH):
    REMOTE_MODEL_URL = "https://github.com/RomiUber/TRACK-D_Traducteur_Langues_Signes/releases/download/model/model_weights.zip"
    ZIP_PATH = "./model_weights.zip"
    EXTRACT_PATH = "./"

    urllib.request.urlretrieve(REMOTE_MODEL_URL, ZIP_PATH)
    print(f"ZIP téléchargé : {os.path.abspath(ZIP_PATH)}")

    # Dézipper
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)

    print(f"Fichiers extraits dans : {os.path.abspath(EXTRACT_PATH)}")
CLASS_INFO = {
    'A': {'desc': 'Poing fermé, pouce sur le côté',  'color': '#818cf8'},
    'B': {'desc': 'Main plate, doigts joints levés', 'color': '#38bdf8'},
    'C': {'desc': 'Main en arc de cercle (forme C)', 'color': '#34d399'},
    'L': {'desc': 'Pouce + index en équerre (L)',    'color': '#fbbf24'},
    'W': {'desc': 'Index, majeur, annulaire levés',  'color': '#f87171'},
    'Y': {'desc': 'Pouce + auriculaire levés (Y)',   'color': '#c084fc'},
}

# ── 1. Config (doit être en premier) ──────────────────────────────
st.set_page_config(
    page_title='ASL Vision — Langue des Signes',
    page_icon='🤟',
    layout='wide'
)

# ── 2. CSS global ─────────────────────────────────────────────────
# Remarque : on utilise UNIQUEMENT st.markdown pour le style.
# Aucun composant JS externe (pas de st.slider) → pas d'erreur localtunnel.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*='css'] {
    font-family: 'Inter', sans-serif !important;
    background: #0d0f1a !important;
    color: #e2e8f0 !important;
}
.main .block-container { padding: 2rem 3rem !important; max-width: 1200px !important; }
#MainMenu, footer, header { visibility: hidden !important; }

/* Sidebar */
section[data-testid='stSidebar'] > div {
    background: #111827 !important;
    border-right: 1px solid rgba(129,140,248,.15) !important;
}

/* Titre principal en dégradé */
h1 {
    font-size: 2rem !important; font-weight: 700 !important;
    background: linear-gradient(135deg,#818cf8,#c084fc) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
h2, h3 { color: #c4b5fd !important; font-weight: 600 !important; }

/* Métrique card */
.m-card {
    background: #1e2132;
    border: 1px solid rgba(129,140,248,.18);
    border-radius: 14px; padding: 1.1rem 1.4rem;
    box-shadow: 0 4px 24px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
}
.m-val  { font-size: 1.9rem; font-weight: 700; color: #a5b4fc; line-height: 1.1; }
.m-lbl  { font-size: .68rem; letter-spacing: 1.5px; text-transform: uppercase; color: #4b5563; margin-top: 3px; }

/* Classe ASL card */
.cls-card {
    background: #1e2132;
    border: 1px solid rgba(255,255,255,.06);
    border-radius: 13px; padding: .9rem .6rem;
    text-align: center;
    transition: transform .2s, border-color .2s;
}
.cls-card:hover { transform: translateY(-4px); border-color: rgba(129,140,248,.4); }
.cls-letter { font-size: 1.6rem; font-weight: 700; margin: 4px 0; }
.cls-desc   { font-size: .7rem; color: #6b7280; line-height: 1.35; }

/* Résultat hero */
.res-hero {
    background: linear-gradient(135deg,#1e1b4b,#0f0e2a);
    border: 1px solid rgba(129,140,248,.4);
    border-radius: 20px; padding: 2rem; text-align: center;
    box-shadow: 0 0 50px rgba(99,102,241,.12), 0 8px 32px rgba(0,0,0,.5);
}
.big-letter {
    font-size: 5.5rem; font-weight: 700; line-height: 1;
    background: linear-gradient(135deg,#a5b4fc,#818cf8,#6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 24px rgba(99,102,241,.55));
}
.conf-badge {
    display: inline-block;
    background: rgba(129,140,248,.12);
    border: 1px solid rgba(129,140,248,.3);
    border-radius: 50px; padding: 5px 16px;
    font-size: 13px; color: #a5b4fc; margin-top: 10px;
}

/* Barres de proba */
.pb-row  { display:flex; align-items:center; gap:10px; margin-bottom:9px; }
.pb-ltr  { font-weight:700; font-size:.95rem; width:20px; color:#c4b5fd; }
.pb-trk  { flex:1; height:7px; background:rgba(255,255,255,.06); border-radius:4px; overflow:hidden; }
.pb-fill { height:100%; border-radius:4px;
           background:linear-gradient(90deg,#6366f1,#a5b4fc);
           box-shadow:0 0 8px rgba(99,102,241,.4); }
.pb-fill.top { background:linear-gradient(90deg,#059669,#34d399); box-shadow:0 0 8px rgba(52,211,153,.4); }
.pb-pct  { font-size:11px; color:#4b5563; width:40px; text-align:right; }
.pb-pct.top { color:#34d399; }

/* Upload zone */
[data-testid='stFileUploader'] {
    background: rgba(99,102,241,.04) !important;
    border: 1.5px dashed rgba(129,140,248,.3) !important;
    border-radius: 14px !important;
}

/* Progress bar */
[data-testid='stProgressBar'] > div > div {
    background: linear-gradient(90deg,#6366f1,#a5b4fc) !important;
}

/* Images */
[data-testid='stImage'] img {
    border-radius: 12px !important;
    border: 1px solid rgba(129,140,248,.18) !important;
}

/* select_slider — retire le slider JS, garde un look propre */
[data-testid='stSlider'] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── 3. Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#374151;text-transform:uppercase">Prototype éducatif</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:19px;font-weight:700;color:#a5b4fc;margin:0 0 16px">🤟 ASL Vision</p>', unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('<p style="font-size:10px;letter-spacing:1.5px;color:#374151;text-transform:uppercase">Modèle</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:13px;color:#c4b5fd;margin:2px 0 12px">Swin-Tiny (Microsoft)</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px;letter-spacing:1.5px;color:#374151;text-transform:uppercase">Dataset</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:13px;color:#c4b5fd;margin:2px 0 12px">ASL Alphabet — Kaggle</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px;letter-spacing:1.5px;color:#374151;text-transform:uppercase">Classes</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:13px;color:#c4b5fd;margin:2px 0 16px">A · B · C · L · W · Y</p>', unsafe_allow_html=True)
    st.markdown('---')
    # select_slider = pas de module JS Slider → pas d'erreur localtunnel
    seuils    = ['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%']
    choix     = st.select_slider('Seuil de confiance', options=seuils, value='60%')
    threshold = int(choix.replace('%','')) / 100
    st.markdown('---')
    with st.expander('Comment ça marche ?'):
        st.markdown('1. **Upload** une photo de main\n2. Swin-Tiny analyse l\'image\n3. Prédiction de la lettre ASL\n4. **GradCAM** montre les zones décisives')
    st.markdown('<p style="font-size:11px;color:#374151;margin-top:16px">⚠️ 6 lettres statiques uniquement.<br>Ne remplace pas un interprète.</p>', unsafe_allow_html=True)

# ── 4. Header ─────────────────────────────────────────────────────
st.title('🤟 Traducteur Langue des Signes ASL')
st.markdown('<p style="color:#4b5563;font-size:14px;margin:-8px 0 24px">Reconnaissance de gestes par Swin-Tiny · Explicabilité GradCAM</p>', unsafe_allow_html=True)

# ── 5. Métriques ──────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [(c1,'93.8%','Accuracy Test'),(c2,'0.937','F1 Macro'),(c3,'0.939','Recall Macro'),(c4,'27.5M','Paramètres')]:
    col.markdown(f'<div class="m-card"><div class="m-val">{val}</div><div class="m-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)

# ── 6. Référence des classes ASL (emojis, pas d'images externes) ──
# On utilise des emojis Unicode au lieu d'URLs Wikimedia
# (les URLs externes ne chargent pas toujours via localtunnel)
EMOJIS = {'A':'✊','B':'🖐','C':'🤌','L':'👆','W':'🤘','Y':'🤙'}
st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#4b5563;text-transform:uppercase;margin-bottom:10px">Gestes reconnus</p>', unsafe_allow_html=True)
rcols = st.columns(6)
for i, cls in enumerate(CLASSES):
    rcols[i].markdown(f"""
    <div class='cls-card'>
        <div style='font-size:2.2rem'>{EMOJIS[cls]}</div>
        <div class='cls-letter' style='color:{CLASS_INFO[cls]["color"]}'>{cls}</div>
        <div class='cls-desc'>{CLASS_INFO[cls]['desc']}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div style="height:32px"></div>', unsafe_allow_html=True)

# ── 7. Modèle ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = AutoModelForImageClassification.from_pretrained(
        MODEL_PATH, num_labels=6, ignore_mismatched_sizes=True)
    m.eval()
    return m

def reshape_transform_swin(tensor, height=7, width=7):
    r = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return r.transpose(2,3).transpose(1,2)

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])
])

# ── 8. Zone principale : st.columns(2) ────────────────────────────
st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#4b5563;text-transform:uppercase;margin-bottom:6px">Image source</p>', unsafe_allow_html=True)
left, right = st.columns(2, gap='large')

with left:
    uploaded = st.file_uploader('', type=['jpg','jpeg','png'], label_visibility='collapsed')
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, use_container_width=True)
        st.markdown(f'<p style="font-size:11px;color:#374151">{uploaded.name} · {uploaded.size//1024} KB · {img.size[0]}×{img.size[1]} px</p>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:3rem;text-align:center;color:#374151;font-size:13px;border:1px dashed rgba(129,140,248,.2);border-radius:14px"><div style="font-size:2rem;margin-bottom:8px">👆</div>Déposez une image de geste ASL</div>', unsafe_allow_html=True)

with right:
    st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#4b5563;text-transform:uppercase;margin-bottom:6px">Résultat</p>', unsafe_allow_html=True)
    if uploaded:
        with st.spinner('Analyse en cours...'):
            model        = load_model()
            t            = val_transform(img).unsqueeze(0)
            with torch.no_grad():
                out      = model(t)
            probs        = torch.softmax(out.logits, dim=-1)[0].numpy()
            pred_idx     = int(np.argmax(probs))
            pred_class   = CLASSES[pred_idx]
            confidence   = float(probs[pred_idx])

        if confidence < threshold:
            st.warning(f'⚠️ Confiance trop faible ({confidence:.1%} < {threshold:.0%}). Essayez une image plus nette.')
        else:
            st.markdown(f"""
            <div class='res-hero'>
                <div class='big-letter'>{pred_class}</div>
                <div class='conf-badge'>{CLASS_INFO[pred_class]['desc']}</div>
                <div style='font-size:13px;color:#6366f1;margin-top:10px'>
                    Confiance · <span style='color:#a5b4fc;font-weight:600'>{confidence:.1%}</span>
                </div>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
            st.progress(confidence)

        # Barres de probabilité
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#4b5563;text-transform:uppercase;margin-bottom:10px">Distribution des probabilités</p>', unsafe_allow_html=True)
        for cls, p in sorted(zip(CLASSES, probs), key=lambda x:-x[1]):
            top = cls == pred_class
            st.markdown(f"""
            <div class='pb-row'>
                <div class='pb-ltr'>{cls}</div>
                <div class='pb-trk'><div class='pb-fill {'top' if top else ''}' style='width:{p*100:.1f}%'></div></div>
                <div class='pb-pct {'top' if top else ''}'>{p*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:3rem;text-align:center;color:#374151;font-size:13px;border:1px dashed rgba(129,140,248,.2);border-radius:14px"><div style="font-size:1.8rem;margin-bottom:8px">🔍</div>La prédiction apparaîtra ici</div>', unsafe_allow_html=True)

# ── 9. GradCAM ────────────────────────────────────────────────────
if uploaded:
    st.markdown('<div style="height:36px"></div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#4b5563;text-transform:uppercase;margin-bottom:10px">Visualisation GradCAM — Explicabilité</p>', unsafe_allow_html=True)
    g1, g2 = st.columns(2, gap='large')
    with g1:
        st.markdown('<p style="font-size:12px;color:#6b7280;margin-bottom:6px">Image originale</p>', unsafe_allow_html=True)
        st.image(img.resize((224,224)), use_container_width=True)
    with g2:
        st.markdown('<p style="font-size:12px;color:#6b7280;margin-bottom:6px">Heatmap GradCAM</p>', unsafe_allow_html=True)
        with st.spinner('Génération GradCAM...'):
            try:
                class W(torch.nn.Module):
                    def __init__(self, m): super().__init__(); self.model = m
                    def forward(self, x): return self.model(x).logits
                wrapped = W(load_model())
                cam     = GradCAM(model=wrapped,
                                  target_layers=[wrapped.model.swin.encoder.layers[-1].blocks[-1].layernorm_after],
                                  reshape_transform=reshape_transform_swin)
                t2      = val_transform(img).unsqueeze(0)
                gc      = cam(input_tensor=t2, targets=[ClassifierOutputTarget(pred_idx)])[0]
                rgb     = np.array(img.resize((224,224))).astype(np.float32)/255.
                overlay = show_cam_on_image(rgb, gc, use_rgb=True)
                st.image(overlay, use_container_width=True)
            except Exception as e:
                st.error(f'Erreur GradCAM : {e}')
    st.markdown('<p style="font-size:12px;color:#374151;margin-top:10px">Zones <b style="color:#f87171">rouges</b> = régions décisives pour la prédiction du modèle.</p>', unsafe_allow_html=True)

# ── 10. Disclaimer ────────────────────────────────────────────────
st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)
st.error('⚠️ Prototype éducatif — 6 lettres ASL statiques uniquement. Ne remplace pas un interprète qualifié.')
