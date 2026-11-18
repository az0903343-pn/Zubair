"""
Budget-Based Vehicle Recommender + Multi-AI Chatbot
Deploy: train on Kaggle -> push to GitHub -> deploy on Streamlit Cloud

API keys (bring your own - I cannot provide real keys):
  - Gemini:   https://aistudio.google.com/app/apikey
  - OpenAI:   https://platform.openai.com/api-keys
  - Groq:     https://console.groq.com/keys

Add them to Streamlit Cloud Secrets (TOML) or a local .env file.
"""
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment (local dev). On Streamlit Cloud, use Secrets instead.
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

try:
    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", GEMINI_KEY)
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", OPENAI_KEY)
    GROQ_KEY = st.secrets.get("GROQ_API_KEY", GROQ_KEY)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vehicle Recommender | Smart Drive AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS: dark animated GCORE-inspired UI (every click is a real button)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    .stApp { background: #070709; color: #e8e8ea; font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    @keyframes fadeInUp {
        from {opacity: 0; transform: translateY(30px);} to {opacity: 1; transform: translateY(0);}
    }
    @keyframes pulseGlow {
        0%,100% {box-shadow: 0 0 20px rgba(255, 110, 0, 0.35);}
        50% {box-shadow: 0 0 40px rgba(255, 110, 0, 0.7);}
    }
    .animate-in { animation: fadeInUp 0.8s cubic-bezier(0.22,1,0.36,1) both; }

    /* Hero */
    .hero { position: relative; padding: 80px 20px 50px; text-align: center; overflow: hidden; }
    .hero::before {
        content: ""; position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 30%, rgba(255,110,0,0.16) 0%, transparent 55%);
        animation: rotate 20s linear infinite; z-index: 0;
    }
    @keyframes rotate { from {transform: rotate(0deg);} to {transform: rotate(360deg);} }
    .hero-content { position: relative; z-index: 2; }
    .hero h1 {
        font-size: 3rem; font-weight: 800; letter-spacing: -1px; margin: 0 0 10px;
        background: linear-gradient(90deg, #fff, #ff7a1a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { font-size: 1.15rem; color: #a8a8b2; margin-bottom: 30px; }

    /* Play button - styled real Streamlit button */
    .play-btn .stButton > button {
        width: 90px; height: 90px; border-radius: 50%; padding: 0;
        background: linear-gradient(135deg, #ff7a1a, #ff4d00);
        animation: pulseGlow 2s infinite; font-size: 1.8rem; display: inline-flex;
        align-items: center; justify-content: center;
    }
    .play-btn .stButton > button:hover { transform: translateY(-2px); }

    /* Cards */
    .card {
        background: #121217; border: 1px solid #24242e; border-radius: 16px;
        padding: 20px; transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        animation: fadeInUp 0.6s both;
    }
    .card:hover { transform: translateY(-6px); border-color: #ff7a1a;
        box-shadow: 0 12px 40px rgba(255,110,0,0.15); }
    .card h4 { margin: 0 0 8px; color: #ff7a1a; font-size: 1.1rem; }
    .card .price { font-size: 1.5rem; font-weight: 800; color: #e8e8ea; }
    .card .meta { color: #9aa0aa; font-size: 0.9rem; margin: 6px 0; }
    .card .advantage {
        display: inline-block; margin-top: 10px; padding: 4px 10px; border-radius: 20px;
        background: #1a1f26; color: #ffd280; font-weight: 700; font-size: 0.85rem; border: 1px solid #333;
    }

    .sec-title {
        font-size: 1.6rem; font-weight: 800; margin: 50px 0 20px;
        display: flex; align-items: center; gap: 10px;
    }
    .sec-title span { width: 8px; height: 24px;
        background: linear-gradient(180deg, #ff7a1a, #ff4d00); border-radius: 4px; }

    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: #121217 !important; color: #e8e8ea !important;
        border: 1px solid #2a2a33 !important; border-radius: 10px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ff7a1a, #ff4d00); color: #fff; border: none;
        border-radius: 10px; padding: 12px 28px; font-weight: 700;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover { transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(255,110,0,0.45); }
    .stButton > button:active { transform: translateY(0); }
    .stButton.secondary > button { background: #1a1f26; border: 1px solid #333; }
    .stButton.secondary > button:hover { background: #24242e; box-shadow: none; }

    /* Chat panel (rendered as a real Streamlit container, fully clickable) */
    .chat-wrap { border: 1px solid #2a2a33; background: #0f1016; border-radius: 16px;
        padding: 12px; margin-top: 10px; animation: fadeInUp 0.5s both; }
    .chat-header { background: linear-gradient(135deg, #ff7a1a, #ff4d00); color: #fff;
        font-weight: 700; padding: 10px 14px; border-radius: 10px; margin-bottom: 10px;
        display: flex; align-items: center; gap: 8px; }
    .msg { max-width: 90%; padding: 10px 14px; border-radius: 12px; font-size: 0.95rem; line-height: 1.5; margin: 6px 0; }
    .msg.user { align-self: flex-end; background: #1a1f26; color: #e8e8ea; border: 1px solid #2a2a33;
        margin-left: auto; border-bottom-right-radius: 2px; }
    .msg.bot { align-self: flex-start; background: #121217; color: #e8e8ea; border: 1px solid #24242e;
        border-bottom-left-radius: 2px; margin-right: auto; }
    .msg .emoji { font-size: 1.4rem; margin-right: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Assalam-o-alaikum! I'm your vehicle assistant 🤖. Ask me about cars, budgets, or comparisons!"}
    ]
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "demo_played" not in st.session_state:
    st.session_state.demo_played = False

# ---------------------------------------------------------------------------
# Load model & data
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = "model/vehicle_model.joblib"
    data_path = "model/vehicle_data.csv"
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None, None
    bundle = joblib.load(model_path)
    df = pd.read_csv(data_path)
    return bundle, df

bundle, df = load_artifacts()

# ---------------------------------------------------------------------------
# Multi-AI chatbot backend: Gemini -> OpenAI -> Groq (fallback chain)
# ---------------------------------------------------------------------------
def chat_with_ai(user_text: str) -> str:
    if GEMINI_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(user_text)
            if resp and getattr(resp, "text", None):
                return resp.text
        except Exception as e:
            st.session_state.setdefault("_ai_errors", []).append(f"Gemini: {e}")
    if OPENAI_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_KEY)
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_text}],
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            st.session_state.setdefault("_ai_errors", []).append(f"OpenAI: {e}")
    if GROQ_KEY:
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_KEY)
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": user_text}],
                temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            st.session_state.setdefault("_ai_errors", []).append(f"Groq: {e}")
    errors = st.session_state.get("_ai_errors", [])
    if errors:
        return ("I'm having trouble reaching the AI backends. Please add valid Gemini / OpenAI / "
                "Groq keys in Streamlit Secrets or .env. Last errors: " + "; ".join(errors[-2:]))
    return ("Please add at least one API key (Gemini, OpenAI, or Groq) to enable the chatbot. "
            "You can still use the vehicle recommender above!")

# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------
def recommend(budget, vtype, fuel, seats):
    if bundle is None or df is None:
        return pd.DataFrame()
    pipe = bundle["pipe"]
    feature_cols = bundle["feature_cols"]
    categorical_cols = bundle["categorical_cols"]

    cand = df[df["Price_USD"] <= budget].copy()
    if vtype != "Any":
        cand = cand[cand["Type"] == vtype]
    if fuel != "Any":
        cand = cand[cand["Fuel_Type"] == fuel]
    if seats is not None:
        cand = cand[cand["Seats"] == int(seats)]

    if cand.empty:
        cand = df[df["Price_USD"] <= budget].copy()
    if cand.empty:
        return pd.DataFrame()

    ideal = pd.DataFrame([{
        "Price_USD": budget,
        "Mileage_kmpl": cand["Mileage_kmpl"].mean(),
        "Seats": cand["Seats"].mean(),
        "Advantage_Score": cand["Advantage_Score"].mean(),
        "Type": cand["Type"].mode()[0] if not cand["Type"].mode().empty else "Sedan",
        "Fuel_Type": cand["Fuel_Type"].mode()[0] if not cand["Fuel_Type"].mode().empty else "Petrol",
    }])
    try:
        _, idx = pipe.kneighbors(ideal[feature_cols + categorical_cols])
        idx = idx[0]
        rec_idx = [i for i in idx if i in cand.index]
        if not rec_idx:
            rec_idx = cand.index.tolist()[:6]
        rec = cand.loc[rec_idx].copy()
    except Exception:
        rec = cand.copy()

    return rec.sort_values("Advantage_Score", ascending=False).head(6)

# ---------------------------------------------------------------------------
# Header / Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="hero"><div class="hero-content">', unsafe_allow_html=True)
st.markdown('<h1>Smart Drive AI</h1>', unsafe_allow_html=True)
st.markdown('<p>Enter your budget and we will recommend the best vehicles you can afford — ranked by advantage, with full comparison.</p>', unsafe_allow_html=True)

# Real, active play button (centered)
pc1, pc2, pc3 = st.columns([1, 1, 1])
with pc2:
    st.markdown('<div class="play-btn">', unsafe_allow_html=True)
    if st.button("▶ Play Demo", key="play_demo", help="Watch entrance animation"):
        st.session_state.demo_played = True
        st.balloons()
        st.success("Demo played! Now enter your budget below to get recommendations.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input form (every click is a real form submit)
# ---------------------------------------------------------------------------
with st.form("budget_form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        budget = st.number_input("Your Budget (USD)", min_value=1000, max_value=200000, value=25000, step=1000)
    with c2:
        vtype = st.selectbox("Vehicle Type", ["Any", "Sedan", "SUV", "Hatchback", "Electric", "MPV", "Off-Roader", "Micro-SUV", "Coupe-SUV", "Crossover"])
    with c3:
        fuel = st.selectbox("Fuel Type", ["Any", "Petrol", "Diesel", "Electric"])
    with c4:
        seats_label = st.selectbox("Seats", ["Any", "4", "5", "6", "7"])
    submitted = st.form_submit_button("Recommend Vehicles 🚀")

seats = None if seats_label == "Any" else int(seats_label)

if submitted:
    st.session_state.recommendations = recommend(budget, vtype, fuel, seats)

# ---------------------------------------------------------------------------
# Recommendations + Comparison
# ---------------------------------------------------------------------------
if st.session_state.recommendations is not None:
    rec = st.session_state.recommendations
    if rec.empty:
        st.warning("No vehicles found within that budget. Try increasing the budget.")
    else:
        st.markdown('<div class="sec-title animate-in"><span></span>Recommended for Your Budget</div>', unsafe_allow_html=True)
        cols = st.columns(min(3, len(rec)))
        for i, (_, row) in enumerate(rec.iterrows()):
            with cols[i % len(cols)]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h4>{row["Name"]}</h4>', unsafe_allow_html=True)
                st.markdown(f'<div class="price">${row["Price_USD"]:,}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="meta">{row["Brand"]} &bull; {row["Type"]} &bull; {row["Fuel_Type"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="meta">{row["Mileage_kmpl"]} kmpl &bull; {row["Seats"]} seats &bull; Safety {row["Safety_Rating"]}/5</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="advantage">Advantage Score: {row["Advantage_Score"]}/100</div>', unsafe_allow_html=True)
                st.caption(row["Description"])
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-title animate-in"><span></span>Side-by-Side Comparison</div>', unsafe_allow_html=True)
        show_cols = ["Name", "Brand", "Type", "Price_USD", "Mileage_kmpl", "Seats", "Fuel_Type", "Safety_Rating", "Advantage_Score"]
        st.dataframe(rec[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=rec["Name"], y=rec["Advantage_Score"], name="Advantage Score",
                             marker_color="#ff7a1a", text=rec["Advantage_Score"], textposition="outside"))
        fig.add_trace(go.Bar(x=rec["Name"], y=rec["Safety_Rating"], name="Safety Rating",
                             marker_color="#2a9df4", text=rec["Safety_Rating"], textposition="outside"))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#070709", plot_bgcolor="#070709",
            font=dict(color="#e8e8ea", family="Inter"), barmode="group", height=420,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chatbot (fully active: real open/close/send buttons)
# ---------------------------------------------------------------------------
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
st.markdown('<div class="chat-header">🤖 Vehicle Assistant</div>', unsafe_allow_html=True)

if not st.session_state.chat_open:
    if st.button("Open Chat", key="open_chat", help="Open the assistant chat"):
        st.session_state.chat_open = True
        st.rerun()
else:
    # Messages
    for m in st.session_state.messages:
        role_class = "user" if m["role"] == "user" else "bot"
        emoji = "🧑" if m["role"] == "user" else "🤖"
        st.markdown(
            f'<div class="msg {role_class}"><span class="emoji">{emoji}</span>{m["content"]}</div>',
            unsafe_allow_html=True,
        )

    user_input = st.chat_input("Type your question...", key="chat_input")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            reply = chat_with_ai(user_input)
        st.session_state.messages.append({"role": "bot", "content": reply})
        st.rerun()

    cclose, _ = st.columns([1, 4])
    with cclose:
        if st.button("✕ Close", key="close_chat", help="Close the chat panel"):
            st.session_state.chat_open = False
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div style="text-align:center;color:#666;padding:40px 20px;font-size:0.85rem;">'
    'Built with Streamlit &bull; Train on Kaggle &bull; Deploy on Streamlit Cloud &bull; '
    'Bring your own Gemini / OpenAI / Groq keys for the chatbot.'
    '</div>',
    unsafe_allow_html=True,
)
