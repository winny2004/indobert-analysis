import streamlit as st
import re
import string
import json
import os
import pandas as pd
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#fafafa',
    'savefig.facecolor': '#ffffff',
    'font.family': 'sans-serif',
    'font.size': 10,
})
import torch
import torch.nn.functional as F
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Analisis Sentimen Review KAI",
    page_icon="🚂",
    layout="wide"
)

BLUE = "#232368"
YELLOW = "#e4640c"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SLANG_PATH = os.path.join(BASE_DIR, 'slang.json')
CSV_PATH = os.path.join(BASE_DIR, 'REVIEW KAI.csv')
IMAGE_PATH = os.path.join(BASE_DIR, 'assets', 'image.png')

DIVIDER_HTML = '<div class="divider"></div>'

load_dotenv(os.path.join(BASE_DIR, '.env'))
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')


@st.cache_resource
def init_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


supabase: Client = init_supabase()


def get_banner_bg_css():
    if not os.path.exists(IMAGE_PATH):
        return ""
    with open(IMAGE_PATH, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    return f"background-image: url('data:image/png;base64,{b64}');"


st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {{
        background-color: #f5f6fa;
        color: #1a1a2e;
        font-family: 'Inter', sans-serif;
    }}

    .stApp > header {{ background-color: transparent; }}

    section[data-testid="stSidebar"] {{
        background-color: {BLUE} !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}

    /* ====== HERO BANNER ====== */
    .hero-banner {{
        position: relative;
        width: 100vw;
        margin-left: calc(-50vw + 50%);
        min-height: 420px;
        {get_banner_bg_css()}
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        border-radius: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }}

    .hero-overlay {{
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(180deg, rgba(0,0,0,0.35) 0%, rgba(0,0,0,0.55) 60%, rgba(35,35,104,0.7) 100%);
        z-index: 1;
    }}

    .hero-content {{
        position: relative;
        z-index: 2;
        width: 100%;
        max-width: 700px;
        padding: 3rem 2rem 2.5rem;
        text-align: center;
    }}

    .hero-title {{
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}

    .hero-subtitle {{
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }}

    /* ====== TEXTAREA ====== */
    div.stTextArea > div > div > textarea {{
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 2px solid #c5c9d6 !important;
        border-radius: 14px !important;
        font-size: 0.95rem !important;
        min-height: 100px !important;
        padding: 1rem 1.1rem !important;
        line-height: 1.6 !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }}

    div.stTextArea > div > div > textarea:focus {{
        border-color: {BLUE} !important;
        box-shadow: 0 0 0 4px rgba(35, 35, 104, 0.12), 0 4px 16px rgba(35, 35, 104, 0.08) !important;
    }}

    div.stTextArea > div > div > textarea::placeholder {{
        color: #b0b4c0 !important;
        font-style: italic !important;
    }}

    div.stTextArea > label {{
        color: #333333 !important;
        font-weight: 600 !important;
    }}

    .input-section-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: #2d2d5e;
        margin-bottom: 0.5rem;
        letter-spacing: -0.2px;
        text-align: center;
    }}

    .input-example {{
        color: #9ca3af;
        font-size: 0.82rem;
        font-style: italic;
        margin-bottom: 0.4rem;
        padding-left: 0.2rem;
    }}

    /* ====== GENERAL ====== */
    .main-title {{
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        color: {BLUE};
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }}

    .subtitle {{
        text-align: center;
        color: #666666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }}

    .divider {{
        height: 3px;
        background: linear-gradient(90deg, {YELLOW}, {BLUE}, {YELLOW});
        border: none;
        margin: 1.5rem 0;
        border-radius: 2px;
    }}

    .section-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: {BLUE};
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }}

    /* ====== RESULT ====== */
    .result-positive {{
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }}

    .result-negative {{
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }}

    .label-positive {{ font-size: 1.6rem; font-weight: 700; color: #22c55e; }}
    .label-negative {{ font-size: 1.6rem; font-weight: 700; color: #ef4444; }}
    .emoji-big {{ font-size: 3rem; margin-bottom: 0.5rem; }}

    /* ====== INFO BOX ====== */
    .info-box {{
        background-color: #ffffff;
        border-left: 4px solid {YELLOW};
        padding: 1rem 1.2rem;
        border-radius: 8px;
        color: #333333;
        font-size: 0.95rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}

    /* ====== REVIEW LIST (no background) ====== */
    .review-link {{
        display: block;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.15rem;
        border-left: 3px solid {BLUE};
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        color: #333333;
    }}

    .review-link:hover {{
        border-left-color: {YELLOW};
        text-decoration: underline;
        text-decoration-color: {YELLOW};
        text-underline-offset: 3px;
    }}

    .review-link-positive {{ border-left-color: #22c55e; }}
    .review-link-negative {{ border-left-color: #ef4444; }}

    .review-text {{
        color: #333333;
        font-size: 0.92rem;
        line-height: 1.5;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }}

    .badge-positive {{
        background-color: #dcfce7;
        color: #22c55e;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }}

    .badge-negative {{
        background-color: #fee2e2;
        color: #ef4444;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }}

    .footer {{
        text-align: center;
        color: #999999;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ====== BUTTONS ====== */
    .stButton {{
        text-align: center !important;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, {BLUE}, #1a1a5e) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: auto !important;
        display: block !important;
        margin: 0 auto !important;
        transition: all 0.3s ease !important;
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, {YELLOW}, #c75a0a) !important;
        box-shadow: 0 0 20px rgba(228, 100, 12, 0.4) !important;
        transform: translateY(-1px) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0px) !important;
    }}

    /* Yellow "Lihat Semua Review" button */
    .btn-yellow-link {{
        display: block;
        text-align: center;
        margin-top: 1rem;
        background: {YELLOW};
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        font-size: 1.05rem;
        width: 100%;
        text-decoration: none;
        transition: all 0.3s ease;
        cursor: pointer;
        box-sizing: border-box;
    }}

    .btn-yellow-link:hover {{
        background: #c75a0a;
        box-shadow: 0 0 16px rgba(228, 100, 12, 0.5);
        transform: translateY(-1px);
        color: #ffffff !important;
        text-decoration: none;
    }}

    /* Dashboard button */
    .btn-dashboard-link {{
        display: block;
        text-align: center;
        margin-top: 1rem;
        background: {BLUE};
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 700;
        font-size: 1.05rem;
        width: 100%;
        text-decoration: none;
        transition: all 0.3s ease;
        cursor: pointer;
        box-sizing: border-box;
    }}

    .btn-dashboard-link:hover {{
        background: #1a1a5e;
        box-shadow: 0 0 16px rgba(35, 35, 104, 0.5);
        transform: translateY(-1px);
        color: #ffffff !important;
        text-decoration: none;
    }}

    /* ====== DETAIL ====== */
    .detail-card {{
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }}

    .detail-label {{
        font-size: 0.85rem;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }}

    .detail-value {{
        font-size: 1rem;
        color: #333;
        line-height: 1.6;
    }}

    /* ====== STAT CARD ====== */
    .stat-card {{
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}

    .stat-number {{
        font-size: 2rem;
        font-weight: 800;
        color: {BLUE};
    }}

    .stat-label {{
        font-size: 0.85rem;
        color: #888;
        font-weight: 500;
    }}

    /* ====== SUCCESS TOAST ====== */
    .success-toast {{
        background-color: #dcfce7;
        border-left: 4px solid #22c55e;
        color: #166534;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }}

    /* ====== RESULT POPUP MODAL ====== */
    .modal-overlay {{
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(4px);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .modal-card {{
        background: #ffffff;
        border-radius: 20px;
        padding: 2.5rem 2.5rem 2rem;
        max-width: 400px;
        width: 90%;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.25);
        text-align: center;
        animation: modalIn 0.35s ease;
    }}

    .modal-card-positive {{
        border-top: 5px solid #22c55e;
    }}

    .modal-card-negative {{
        border-top: 5px solid #ef4444;
    }}

    .modal-label {{
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.3px;
        margin-bottom: 0.3rem;
    }}

    .modal-label-positive {{ color: #16a34a; }}
    .modal-label-negative {{ color: #dc2626; }}

    .modal-subtext {{
        font-size: 0.92rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }}

    .modal-close-btn {{
        display: inline-block;
        padding: 0.6rem 2.5rem;
        border-radius: 10px;
        border: 2px solid #333333;
        background: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
        color: #333333;
        cursor: pointer;
        transition: all 0.25s ease;
        text-decoration: none;
    }}

    .modal-close-btn:hover {{
        background: #333333;
        color: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }}

    @keyframes modalIn {{
        from {{ opacity: 0; transform: scale(0.92) translateY(15px); }}
        to {{ opacity: 1; transform: scale(1) translateY(0); }}
    }}

    /* ====== DASHBOARD ====== */
    .dashboard-stat {{
        background: #ffffff;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-top: 4px solid {BLUE};
    }}

    .dashboard-stat-number {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {BLUE};
    }}

    .dashboard-stat-label {{
        font-size: 0.85rem;
        color: #888;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }}

    .chart-card {{
        background: #ffffff;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e8e8ee;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }}

    .chart-center-wrapper {{
        display: flex;
        justify-content: center;
    }}

    .chart-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {BLUE};
        margin-bottom: 1rem;
    }}

    .filter-label {{
        font-size: 0.9rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.3rem;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_slang():
    if os.path.exists(SLANG_PATH):
        with open(SLANG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


HF_MODEL_ID = "marcovanbasten/indobert-analysis"
LABEL_MAP = {0: "Tidak Puas", 1: "Puas"}


@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model


@st.cache_data
def load_reviews():
    df = pd.read_csv(CSV_PATH, sep=';')
    return df


def case_folding(text):
    return text.lower()


def cleaning(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text


def normalisasi_kata(text, slang_dict):
    words = text.split()
    return ' '.join([slang_dict.get(word, word) for word in words])


def preprocess_text(text, slang_dict):
    text = case_folding(text)
    text = cleaning(text)
    text = normalisasi_kata(text, slang_dict)
    return text


def predict_sentiment(text, tokenizer, model, slang_dict):
    processed = preprocess_text(text, slang_dict)
    device = next(model.parameters()).device
    encoding = tokenizer(
        processed, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1).squeeze()
    pred_id = torch.argmax(probs).item()
    prediction = LABEL_MAP[pred_id]
    prob_dict = {
        "Puas": probs[1].item(),
        "Tidak Puas": probs[0].item()
    }
    return prediction, processed, prob_dict


def init_session_reviews():
    if 'reviews' not in st.session_state:
        st.session_state.reviews = []
        if supabase:
            try:
                resp = supabase.table('reviews').select('*').order('created_at', desc=True).limit(50).execute()
                for row in resp.data:
                    st.session_state.reviews.append({
                        'id': row['id'],
                        'text': row['review_text'],
                        'prediction': row['prediction'],
                        'prob_puas': row['prob_puas'],
                        'prob_tidak': row['prob_tidak_puas'],
                        'processed': row['processed_text'] or '',
                        'timestamp': datetime.fromisoformat(row['created_at'].replace('Z', '+00:00')).strftime('%d/%m/%Y %H:%M'),
                    })
            except Exception:
                pass


def add_review_to_history(text, prediction, prob_dict, processed):
    prob_puas = prob_dict.get('Puas', 0) * 100
    prob_tidak = prob_dict.get('Tidak Puas', 0) * 100
    db_id = None

    if supabase:
        try:
            resp = supabase.table('reviews').insert({
                'review_text': text,
                'processed_text': processed,
                'prediction': prediction,
                'prob_puas': prob_puas,
                'prob_tidak_puas': prob_tidak,
            }).execute()
            if resp.data:
                db_id = resp.data[0]['id']
        except Exception:
            db_id = len(st.session_state.reviews) + 1

    if db_id is None:
        db_id = len(st.session_state.reviews) + 1

    entry = {
        'id': db_id,
        'text': text,
        'prediction': prediction,
        'prob_puas': prob_puas,
        'prob_tidak': prob_tidak,
        'processed': processed,
        'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M'),
    }
    st.session_state.reviews.insert(0, entry)


init_session_reviews()
slang_dict = load_slang()
tokenizer, indobert_model = load_models()
df_reviews = load_reviews()


# ==================== PAGE ROUTING ====================
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_review_id' not in st.session_state:
    st.session_state.selected_review_id = None

# Handle query param navigation from anchor clicks
qp = st.query_params
if 'close_modal' in qp:
    if 'last_result' in st.session_state:
        del st.session_state.last_result
    del st.query_params['close_modal']
elif 'detail' in qp:
    try:
        detail_id = int(qp['detail'])
        source = qp.get('source', 'home')
        st.session_state.selected_review_id = detail_id
        st.session_state.detail_source = source
        st.session_state.page = 'detail'
        del st.query_params['detail']
        if 'source' in st.query_params:
            del st.query_params['source']
    except (ValueError, TypeError):
        pass
elif 'page' in qp:
    target = qp['page']
    if target == 'history':
        st.session_state.page = 'history'
    elif target == 'dashboard':
        st.session_state.page = 'dashboard'
    del st.query_params['page']


def go_home():
    st.session_state.page = 'home'
    st.rerun()


def go_history():
    st.session_state.page = 'history'
    st.rerun()


def go_dashboard():
    st.session_state.page = 'dashboard'
    st.rerun()


def go_detail(review_id, source='home'):
    st.session_state.selected_review_id = review_id
    st.session_state.detail_source = source
    st.session_state.page = 'detail'
    st.rerun()


# ==================== HOME PAGE ====================
if st.session_state.page == 'home':
    # HERO BANNER full-width (di luar columns)
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-title">ANALISIS SENTIMEN</div>
            <div class="hero-subtitle">Review Kereta Api Indonesia &mdash; Puas atau Tidak Puas?</div>
            <div style="color:rgba(255,255,255,0.6); font-size:0.85rem; margin-top:0.3rem;">Analisis sentimen otomatis menggunakan Machine Learning untuk mengetahui kepuasan penumpang kereta api berdasarkan review.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 6, 1])
    with col_center:
        # Textarea on top of banner (inside the hero visual context)
        st.markdown('<div style="margin-top:-3.5rem; margin-bottom:1rem; position:relative; z-index:5;">', unsafe_allow_html=True)

        st.markdown('<div class="input-section-title">Masukkan Ulasan Untuk Melihat Hasil Analisis Sentimen</div>', unsafe_allow_html=True)

        review_input = st.text_area(
            "Tulis review tentang pengalaman kereta api Anda:",
            placeholder="Contoh: pelayanannya ramah dan menyenangkan, kereta bersih dan nyaman...",
            label_visibility="collapsed",
            height=100,
            key="review_input_area"
        )

        if st.button("Analisis Sentimen", key="analyze_btn"):
            if review_input.strip():
                with st.spinner("Memproses..."):
                    prediction, processed, prob_dict = predict_sentiment(
                        review_input, tokenizer, indobert_model, slang_dict
                    )
                    add_review_to_history(review_input, prediction, prob_dict, processed)
                    st.session_state.last_result = {
                        'prediction': prediction,
                        'prob_dict': prob_dict,
                    }
            else:
                st.warning("Silakan masukkan review terlebih dahulu.")

        if 'last_result' in st.session_state:
            res = st.session_state.last_result
            is_positive = res['prediction'] == "Puas"
            card_cls = "modal-card-positive" if is_positive else "modal-card-negative"
            label_cls = "modal-label-positive" if is_positive else "modal-label-negative"
            emoji = "&#x1F604;" if is_positive else "&#x1F61E;"

            st.markdown(f"""
            <div class="modal-overlay" id="resultModal">
                <div class="modal-card {card_cls}">
                    <div class="emoji-big">{emoji}</div>
                    <div class="modal-label {label_cls}">{res['prediction']}</div>
                    <form action="" method="get">
                        <button type="submit" name="close_modal" value="1" class="modal-close-btn">Tutup</button>
                    </form>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # 5 Review Terakhir
        st.markdown(f'<div class="section-title">Review Terakhir</div>', unsafe_allow_html=True)

        recent_reviews = st.session_state.reviews[:5]
        if recent_reviews:
            for rev in recent_reviews:
                badge_class = 'badge-positive' if rev['prediction'] == 'Puas' else 'badge-negative'
                link_class = 'review-link-positive' if rev['prediction'] == 'Puas' else 'review-link-negative'
                truncated = rev['text'][:80]
                if len(rev['text']) > 80:
                    truncated += "..."
                st.markdown(f"""
                <a href="?detail={rev['id']}&source=home" target="_self" class="review-link {link_class}" style="text-decoration:none;">
                    <div class="review-text">{truncated}</div>
                    <div style="margin-top:0.3rem;">
                        <span class="{badge_class}">{rev['prediction']}</span>
                        <span style="color:#999; font-size:0.8rem; margin-left:0.5rem;">{rev['timestamp']}</span>
                    </div>
                </a>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding:1rem; text-align:center; color:#999; font-style:italic;">
                Belum ada review. Mulai analisis sentimen pertama Anda!
            </div>
            """, unsafe_allow_html=True)

        # Buttons: Lihat Semua Review & Lihat Dashboard
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.markdown(f"""
            <a href="?page=history" target="_self" class="btn-yellow-link">
                Lihat Semua Review
            </a>
            """, unsafe_allow_html=True)
        with btn_col2:
            st.markdown(f"""
            <a href="?page=dashboard" target="_self" class="btn-dashboard-link">
                Lihat Dashboard
            </a>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="footer">
            Model: IndoBERT (indobenchmark/indobert-base-p1) &bull; Data: Review KAI<br>
            Dibuat dengan Streamlit
        </div>
        """, unsafe_allow_html=True)


# ==================== HISTORY PAGE ====================
elif st.session_state.page == 'history':
    col_left, col_center, col_right = st.columns([1, 6, 1])
    with col_center:
        if st.button("Kembali ke Beranda", key="back_home_history"):
            go_home()

        st.markdown(f'<div class="main-title">Semua Review</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Riwayat analisis sentimen yang telah dilakukan</div>', unsafe_allow_html=True)
        st.markdown(DIVIDER_HTML, unsafe_allow_html=True)

        all_reviews = st.session_state.reviews
        if all_reviews:
            total = len(all_reviews)
            puas_count = sum(1 for r in all_reviews if r['prediction'] == 'Puas')
            tidak_count = total - puas_count

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{total}</div>
                    <div class="stat-label">Total Review</div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color:#22c55e;">{puas_count}</div>
                    <div class="stat-label">Puas</div>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color:#ef4444;">{tidak_count}</div>
                    <div class="stat-label">Tidak Puas</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

            for rev in all_reviews:
                badge_class = 'badge-positive' if rev['prediction'] == 'Puas' else 'badge-negative'
                link_class = 'review-link-positive' if rev['prediction'] == 'Puas' else 'review-link-negative'
                truncated = rev['text'][:100]
                if len(rev['text']) > 100:
                    truncated += "..."
                st.markdown(f"""
                <a href="?detail={rev['id']}&source=history" target="_self" class="review-link {link_class}" style="text-decoration:none;">
                    <div class="review-text">{truncated}</div>
                    <div style="margin-top:0.3rem;">
                        <span class="{badge_class}">{rev['prediction']}</span>
                        <span style="color:#999; font-size:0.8rem; margin-left:0.5rem;">{rev['timestamp']}</span>
                        <span style="color:#999; font-size:0.8rem; margin-left:0.5rem;">Puas: {rev['prob_puas']:.1f}%</span>
                    </div>
                </a>
                """, unsafe_allow_html=True)
        else:
            st.info("Belum ada review yang dianalisis. Kembali ke beranda untuk mulai analisis.")


# ==================== DASHBOARD PAGE ====================
elif st.session_state.page == 'dashboard':
    col_left, col_center, col_right = st.columns([1, 6, 1])
    with col_center:
        if st.button("Kembali ke Beranda", key="back_home_dashboard"):
            go_home()

        st.markdown(f'<div class="main-title">Dashboard Sentimen</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Ringkasan hasil analisis sentimen review KAI</div>', unsafe_allow_html=True)
        st.markdown(DIVIDER_HTML, unsafe_allow_html=True)

        all_reviews = st.session_state.reviews

        if all_reviews:
            import datetime as _dt

            puas_count = sum(1 for r in all_reviews if r['prediction'] == 'Puas')
            tidak_count = len(all_reviews) - puas_count

            # Parse timestamps to extract month
            for r in all_reviews:
                if 'month_key' not in r:
                    try:
                        ts = _dt.datetime.strptime(r['timestamp'], '%d/%m/%Y %H:%M')
                        r['month_key'] = ts.strftime('%Y-%m')
                        r['month_label'] = ts.strftime('%B %Y')
                    except (ValueError, KeyError):
                        r['month_key'] = 'unknown'
                        r['month_label'] = 'Tidak Diketahui'

            months_available = sorted(set(r['month_key'] for r in all_reviews if r.get('month_key') != 'unknown'), reverse=True)
            month_labels = {}
            for m in months_available:
                try:
                    parsed = _dt.datetime.strptime(m, '%Y-%m')
                    month_labels[m] = parsed.strftime('%B %Y')
                except ValueError:
                    month_labels[m] = m

            # Filter per bulan
            st.markdown('<div class="filter-label">Filter Berdasarkan Bulan</div>', unsafe_allow_html=True)
            filter_options = ['Semua Bulan'] + [month_labels[m] for m in months_available]
            selected_filter = st.selectbox("Pilih bulan:", filter_options, label_visibility="collapsed", key="month_filter")

            if selected_filter == 'Semua Bulan':
                filtered_reviews = all_reviews
            else:
                selected_month = [k for k, v in month_labels.items() if v == selected_filter][0]
                filtered_reviews = [r for r in all_reviews if r.get('month_key') == selected_month]

            f_puas = sum(1 for r in filtered_reviews if r['prediction'] == 'Puas')
            f_tidak = len(filtered_reviews) - f_puas

            # Bar chart: perbandingan Puas vs Tidak Puas (di atas stat cards)
            st.markdown(f"""
            <div class="chart-card">
                <div class="chart-title" style="font-size:0.95rem; margin-bottom:0.5rem;">Perbandingan Sentimen{' - ' + selected_filter if selected_filter != 'Semua Bulan' else ''}</div>
            """, unsafe_allow_html=True)

            fig1, ax1 = plt.subplots(figsize=(7, 2.8))
            fig1.patch.set_facecolor('#ffffff')
            ax1.set_facecolor('#fafafa')
            ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#e0e0e0')
            ax1.set_axisbelow(True)

            bar_colors = ['#22c55e', '#ef4444']
            bar_labels_list = ['Puas', 'Tidak Puas']
            bar_values = [f_puas, f_tidak]
            bars = ax1.bar([0.35, 0.65], bar_values, color=bar_colors, width=0.12, edgecolor='#ffffff', linewidth=1, zorder=3)

            for bar, val in zip(bars, bar_values):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                         str(val), ha='center', va='bottom', fontweight='700', fontsize=10, color='#333333')

            ax1.set_xticks([0.35, 0.65])
            ax1.set_xticklabels(bar_labels_list, fontsize=9, color='#666666')
            ax1.set_xlim(0, 1)
            ax1.set_ylabel('Jumlah', fontsize=8, fontweight='600', color='#888888')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('#cccccc')
            ax1.spines['bottom'].set_color('#cccccc')
            ax1.tick_params(axis='both', colors='#888888', labelsize=8)
            ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

            max_y = max(bar_values) if bar_values else 1
            ax1.set_ylim(0, max_y + max(1, max_y * 0.25))

            fig1.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

            # Stat cards (di bawah chart)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f"""
                <div class="dashboard-stat" style="padding:1rem;">
                    <div class="dashboard-stat-number" style="font-size:1.6rem;">{len(filtered_reviews)}</div>
                    <div class="dashboard-stat-label" style="font-size:0.75rem;">Total Review</div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div class="dashboard-stat" style="border-top-color:#22c55e; padding:1rem;">
                    <div class="dashboard-stat-number" style="color:#22c55e; font-size:1.6rem;">{f_puas}</div>
                    <div class="dashboard-stat-label" style="font-size:0.75rem;">Puas</div>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""
                <div class="dashboard-stat" style="border-top-color:#ef4444; padding:1rem;">
                    <div class="dashboard-stat-number" style="color:#ef4444; font-size:1.6rem;">{f_tidak}</div>
                    <div class="dashboard-stat-label" style="font-size:0.75rem;">Tidak Puas</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.info("Belum ada review yang dianalisis. Kembali ke beranda untuk mulai analisis.")


# ==================== DETAIL PAGE ====================
elif st.session_state.page == 'detail':
    col_left, col_center, col_right = st.columns([1, 6, 1])
    with col_center:
        source_page = st.session_state.get('detail_source', 'home')
        if st.button("Kembali", key="back_from_detail"):
            if source_page == 'history':
                go_history()
            else:
                go_home()

        review_id = st.session_state.selected_review_id
        review_data = None
        for r in st.session_state.reviews:
            if r['id'] == review_id:
                review_data = r
                break

        if review_data:
            badge_class = 'badge-positive' if review_data['prediction'] == 'Puas' else 'badge-negative'

            if review_data['prediction'] == "Puas":
                st.markdown("""
                <div class="result-positive">
                    <div class="emoji-big">&#x1F604;</div>
                    <div class="label-positive">Puas</div>
                    <div style="color:#666666; margin-top:0.5rem;">Review ini menunjukkan kepuasan terhadap layanan kereta api.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-negative">
                    <div class="emoji-big">&#x1F61E;</div>
                    <div class="label-negative">Tidak Puas</div>
                    <div style="color:#666666; margin-top:0.5rem;">Review ini menunjukkan ketidakpuasan terhadap layanan kereta api.</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="detail-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                    <span class="{badge_class}" style="font-size:0.9rem; padding:4px 14px;">{review_data['prediction']}</span>
                    <span style="color:#999; font-size:0.85rem;">{review_data['timestamp']}</span>
                </div>
                <div class="detail-label">Review Asli</div>
                <div class="detail-value" style="margin-bottom:1.2rem;">{review_data['text']}</div>
                <div class="detail-label">Teks Setelah Preprocessing</div>
                <div class="detail-value" style="background:#f5f6fa; padding:0.8rem; border-radius:8px; font-family:monospace; font-size:0.9rem;">{review_data['processed']}</div>
            </div>
            """, unsafe_allow_html=True)

            pc1, pc2 = st.columns(2)
            with pc1:
                st.markdown(f"""
                <div class="info-box">
                    <strong style="color:#22c55e;">Probabilitas Puas</strong><br>
                    <span style="font-size:1.8rem; font-weight:700; color:#22c55e;">{review_data['prob_puas']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            with pc2:
                st.markdown(f"""
                <div class="info-box">
                    <strong style="color:#ef4444;">Probabilitas Tidak Puas</strong><br>
                    <span style="font-size:1.8rem; font-weight:700; color:#ef4444;">{review_data['prob_tidak']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Review tidak ditemukan.")
