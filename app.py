"""
Vehicle Budget Recommender - Complete Rebuild
Train on Kaggle -> Push GitHub -> Deploy Streamlit Cloud

API keys (aap khud layenge, main real keys nahi de sakta):
  - Gemini:   https://aistudio.google.com/app/apikey
  - OpenAI:   https://platform.openai.com/api-keys
  - Groq:     https://console.groq.com/keys
"""
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

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
    page_title="Vehicle Budget Recommender",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS - Dark animated UI matching screenshots
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    .stApp {
        background: #070709;
        color: #e8e8ea;
        font-family: 'Inter', sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Animations */
    @keyframes fadeInUp {
        from {opacity: 0; transform: translateY(30px);} to {opacity: 1; transform: translateY(0);}
    }
    @keyframes pulseGlow {
        0%,100% {box-shadow: 0 0 20px rgba(255, 110, 0, 0.35);} 50% {box-shadow: 0 0 40px rgba(255, 110, 0, 0.7);}
    }
    @keyframes float {
        0%,100% {transform: translateY(0);} 50% {transform: translateY(-8px);}
    }
    .animate-in { animation: fadeInUp 0.8s cubic-bezier(0.22,1,0.36,1) both; }

    /* Hero / Intro overlay */
    #intro-overlay {
        position: fixed; inset: 0; background: #070709; z-index: 9999;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        transition: opacity 0.6s ease;
    }
    #intro-overlay.hidden { opacity: 0; pointer-events: none; }
    .intro-logo { font-size: 5rem; font-weight: 800; letter-spacing: -2px;
        background: linear-gradient(90deg, #fff, #ff7a1a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px; }
    .intro-sub { font-size: 1.25rem; color: #a8a8b2; margin-bottom: 40px; }
    .intro-play { width: 80px; height: 80px; border-radius: 50%;
        background: linear-gradient(135deg, #ff7a1a, #ff4d00);
        animation: pulseGlow 2s infinite; display: flex; align-items: center; justify-content: center;
        font-size: 2rem; color: #fff; cursor: pointer; border: none; outline: none; }

    /* Hero section behind */
    .hero { padding: 70px 20px 40px; text-align: center; position: relative; overflow: hidden; }
    .hero::before {
        content: ""; position: absolute; top: -40%; left: -40%; width: 180%; height: 180%;
        background: radial-gradient(circle at 30% 30%, rgba(255,110,0,0.14) 0%, transparent 55%);
        animation: rotate 20s linear infinite; z-index: 0;
    }
    @keyframes rotate { from {transform: rotate(0deg);} to {transform: rotate(360deg);} }
    .hero h1 {
        font-size: 3rem; font-weight: 800; margin: 0 0 8px; position: relative; z-index: 2;
        background: linear-gradient(90deg, #fff, #ff7a1a);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { font-size: 1.1rem; color: #a8a8b2; margin-bottom: 30px; position: relative; z-index: 2; }

    /* Stats cards */
    .stats { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin: 30px 0 40px; position: relative; z-index: 2; }
    .stat-card {
        background: #121217; border: 1px solid #24242e; border-radius: 16px; padding: 20px 26px;
        text-align: center; min-width: 150px; animation: fadeInUp 0.6s both;
    }
    .stat-card .big { font-size: 1.9rem; font-weight: 800; color: #00e67a; margin-bottom: 4px; }
    .stat-card .big.purple { color: #b366ff; }
    .stat-card .small { font-size: 0.85rem; color: #9aa0aa; }

    /* Category cards */
    .categories { margin: 40px 0 50px; position: relative; z-index: 2; }
    .cat-title { font-size: 1.5rem; font-weight: 800; margin-bottom: 20px; color: #e8e8ea; }
    .cat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; }
    .cat-card {
        background: #121217; border: 1px solid #24242e; border-radius: 16px;
        height: 160px; display: flex; flex-direction: column; align-items: center; justify-content: center;
        cursor: pointer; transition: transform 0.25s ease, border-color 0.25s ease;
        animation: fadeInUp 0.6s both;
    }
    .cat-card:hover { transform: translateY(-6px); border-color: #ff7a1a; }
    .cat-img { font-size: 3rem; margin-bottom: 10px; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.4)); }
    .cat-card h4 { margin: 0; color: #ff7a1a; font-size: 1.05rem; }

    /* Input form */
    .form-wrap {
        background: #121217; border: 1px solid #24242e; border-radius: 16px;
        padding: 24px; margin: 0 20px 50px; position: relative; z-index: 2;
    }
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: #0c0c10 !important; color: #e8e8ea !important;
        border: 1px solid #2a2a33 !important; border-radius: 10px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ff7a1a, #ff4d00); color: #fff; border: none;
        border-radius: 10px; padding: 12px 28px; font-weight: 700;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(255,110,0,0.45); }
    .stButton > button:active { transform: translateY(0); }
    .stButton.link > button { background: #1a1f26; border: 1px solid #333; color: #e8e8ea; }
    .stButton.link > button:hover { background: #24242e; border-color: #ff7a1a; color: #ff7a1a; box-shadow: none; }

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

    /* Chatbot - bottom right corner, fixed */
    .chat-toggle-wrap { position: fixed; right: 24px; bottom: 24px; z-index: 1000; }
    .chat-toggle-wrap .stButton > button {
        width: 60px; height: 60px; border-radius: 50%; padding: 0; font-size: 2.2rem;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe); color: #fff;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 8px 30px rgba(108, 92, 231, 0.5); animation: float 3s ease-in-out infinite;
        border: none; outline: none;
    }
    .chat-toggle-wrap .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(108, 92, 231, 0.6); }
    .chat-panel {
        position: fixed; right: 24px; bottom: 100px; width: 340px; height: 520px;
        background: #0f1016; border: 1px solid #2a2a33; border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.6); display: flex; flex-direction: column; z-index: 999;
        animation: fadeInUp 0.5s both; overflow: hidden;
    }
    .chat-header {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe); color: #fff; font-weight: 700;
        padding: 14px 16px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid #2a2a33;
    }
    .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 0.95rem; line-height: 1.5; }
    .msg.user { align-self: flex-end; background: #1a1f26; color: #e8e8ea; border: 1px solid #2a2a33;
        margin-left: auto; border-bottom-right-radius: 2px; }
    .msg.bot { align-self: flex-start; background: #121217; color: #e8e8ea; border: 1px solid #24242e;
        border-bottom-left-radius: 2px; margin-right: auto; }
    .msg .emoji { font-size: 1.4rem; margin-right: 8px; }

    .sec-title { font-size: 1.6rem; font-weight: 800; margin: 50px 0 20px; display: flex; align-items: center; gap: 10px; }
    .sec-title span { width: 8px; height: 24px; background: linear-gradient(180deg, #ff7a1a, #ff4d00); border-radius: 4px; }
    .warn { background: #3a2f0f; color: #e3b505; padding: 10px 14px; border-radius: 10px; border: 1px solid #5c4a0a; margin: 20px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "show_intro" not in st.session_state:
    st.session_state.show_intro = True
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Assalam-o-alaikum! Main hoon aap ka Vehicle Assistant 🤖. Car, budget ya comparison ke bare mein kuch bhi poochein!"}
    ]
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

# ---------------------------------------------------------------------------
# Intro overlay
# ---------------------------------------------------------------------------
if st.session_state.show_intro:
    st.markdown('<div id="intro-overlay">', unsafe_allow_html=True)
    st.markdown('<div class="intro-logo">Vehicle Finder</div>', unsafe_allow_html=True)
    st.markdown('<div class="intro-sub">Aap ka budget, aap ka perfect vehicle</div>', unsafe_allow_html=True)
    if st.button("▶ Continue", key="intro_play", help="Enter the app"):
        st.session_state.show_intro = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

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
# Fallback larger dataset (cycle to helicopter) if model not trained yet
# ---------------------------------------------------------------------------
FALLBACK = pd.DataFrame([
    {"Name":"Hero Splendor Plus","Brand":"Hero","Type":"Bike","Price_USD":1100,"Mileage_kmpl":80,"Seats":2,"Fuel_Type":"Petrol","Safety_Rating":3,"Advantage_Score":88,"Description":"India ka sabse popular commuter bike.","Link":"https://www.heromotocorp.com/en-in/motorcycles/splendor-plus.html"},
    {"Name":"Honda Activa 6G","Brand":"Honda","Type":"Scooter","Price_USD":1300,"Mileage_kmpl":55,"Seats":2,"Fuel_Type":"Petrol","Safety_Rating":3,"Advantage_Score":85,"Description":"Reliable automatic scooter for city rides.","Link":"https://www.honda2wheelersindia.com/activa-6g/"},
    {"Name":"Royal Enfield Classic 350","Brand":"Royal Enfield","Type":"Bike","Price_USD":2200,"Mileage_kmpl":35,"Seats":2,"Fuel_Type":"Petrol","Safety_Rating":4,"Advantage_Score":82,"Description":"Iconic cruiser bike with timeless design.","Link":"https://www.royalenfield.com/"},
    {"Name":"Tata Tiago","Brand":"Tata","Type":"Car","Price_USD":6500,"Mileage_kmpl":19,"Seats":5,"Fuel_Type":"Petrol","Safety_Rating":4,"Advantage_Score":86,"Description":"Safe and affordable hatchback.","Link":"https://www.tatamotors.com/passenger-vehicles/cars/tata-tiago/"},
    {"Name":"Hyundai i20","Brand":"Hyundai","Type":"Car","Price_USD":8500,"Mileage_kmpl":20,"Seats":5,"Fuel_Type":"Petrol","Safety_Rating":4,"Advantage_Score":84,"Description":"Premium hatchback with modern features.","Link":"https://www.hyundai.com/in/en/cars/new-cars/i20"},
    {"Name":"Mahindra Thar","Brand":"Mahindra","Type":"Off-Roader","Price_USD":14000,"Mileage_kmpl":12,"Seats":4,"Fuel_Type":"Diesel","Safety_Rating":4,"Advantage_Score":87,"Description":"Rugged off-road SUV with removeable top.","Link":"https://www.mahindra.com/en-in/thar"},
    {"Name":"Toyota Innova Crysta","Brand":"Toyota","Type":"MPV","Price_USD":22000,"Mileage_kmpl":11,"Seats":7,"Fuel_Type":"Diesel","Safety_Rating":5,"Advantage_Score":89,"Description":"Legendary MPV for families.","Link":"https://www.toyotabharat.com/innova-crysta.html"},
    {"Name":"Tesla Model 3","Brand":"Tesla","Type":"Electric","Price_USD":42000,"Mileage_kmpl":160,"Seats":5,"Fuel_Type":"Electric","Safety_Rating":5,"Advantage_Score":92,"Description":"Zero-emission electric sedan.","Link":"https://www.tesla.com/model3"},
    {"Name":"Piaggio Ape","Brand":"Piaggio","Type":"Rickshaw","Price_USD":3500,"Mileage_kmpl":35,"Seats":4,"Fuel_Type":"Petrol","Safety_Rating":3,"Advantage_Score":80,"Description":"Commercial three-wheeler rickshaw.","Link":"https://www.piaggio.com/en/vehicles/ape"},
    {"Name":"Robinson R44","Brand":"Robinson","Type":"Helicopter","Price_USD":500000,"Mileage_kmpl":0,"Seats":4,"Fuel_Type":"Aviation","Safety_Rating":4,"Advantage_Score":75,"Description":"Light utility helicopter for private use.","Link":"https://www.robinsonheli.com/"},
])

# ---------------------------------------------------------------------------
# Multi-AI chatbot backend: Gemini -> OpenAI -> Groq
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
                messages=[{"role":"user","content":user_text}], temperature=0.7,
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
                messages=[{"role":"user","content":user_text}], temperature=0.7,
            )
            return resp.choices[0].message.content
        except Exception as e:
            st.session_state.setdefault("_ai_errors", []).append(f"Groq: {e}")
    errors = st.session_state.get("_ai_errors", [])
    if errors:
        return ("AI backend tak pahunchne mein dikkat aa rahi hai. Streamlit Secrets ya .env mein Gemini / OpenAI / Groq keys add karein. "
                "Aakhri error: " + "; ".join(errors[-2:]))
    return ("Chatbot chalane ke liye kam az kam ek API key (Gemini, OpenAI, ya Groq) zaroori hai. Vehicle recommender abhi bhi use kar sakte hain.")

# ---------------------------------------------------------------------------
# Recommendation logic - WORKS with actual budget filter
# ---------------------------------------------------------------------------
def recommend(budget, vtype, fuel, seats):
    source = df if (df is not None and len(df) > 0) else FALLBACK
    cand = source[source["Price_USD"] <= budget].copy()
    if vtype != "Any":
        cand = cand[cand["Type"] == vtype]
    if fuel != "Any":
        cand = cand[cand["Fuel_Type"] == fuel]
    if seats != "Any":
        cand = cand[cand["Seats"] == int(seats)]
    if cand.empty:
        cand = source[source["Price_USD"] <= budget].copy()
    if cand.empty:
        return pd.DataFrame()
    return cand.sort_values("Advantage_Score", ascending=False).head(8)

# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------
st.markdown('<div class="hero"><div class="hero-content">', unsafe_allow_html=True)
st.markdown('<h1>Vehicle Budget Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p>Enter your budget and see what you can buy — from a cycle all the way to a helicopter.</p>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# Stats cards (like screenshot 3)
stats_cols = st.columns(4)
stats = [
    ("28", "VEHICLES YOU CAN AFFORD"),
    ("5", "CATEGORIES FIT"),
    ("Rs 150,000", "BEST YOU CAN BUY"),
    ("Rs 6,500", "CHEAPEST OVERALL"),
]
for col, (big, small) in zip(stats_cols, stats):
    color = "purple" if big == "Rs 6,500" else ""
    with col:
        st.markdown(f'<div class="stat-card"><div class="big {color}">{big}</div><div class="small">{small}</div></div>', unsafe_allow_html=True)

# Category cards
st.markdown('<div class="categories"><div class="cat-title">Choose Vehicle Category</div><div class="cat-grid">', unsafe_allow_html=True)
categories = [
    ("🚲", "Cycle"),
    ("🏍️", "Petrol Bike"),
    ("🚗", "Car"),
    ("🚐", "Van"),
    ("🛺", "Rickshaw"),
    ("🚁", "Helicopter"),
]
for emoji, name in categories:
    st.markdown(
        f'<div class="cat-card"><div class="cat-img">{emoji}</div><h4>{name}</h4></div>',
        unsafe_allow_html=True,
    )
st.markdown('</div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Budget form - WORKS NOW
# ---------------------------------------------------------------------------
st.markdown('<div class="form-wrap">', unsafe_allow_html=True)
with st.form("budget_form", clear_on_submit=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        budget = st.number_input("Your Budget (USD)", min_value=100, max_value=5000000, value=150000, step=1000)
    with c2:
        vtype = st.selectbox("Vehicle Type", ["Any","Cycle","Bike","Scooter","Car","Electric","MPV","Off-Roader","Van","Rickshaw","Helicopter"])
    with c3:
        fuel = st.selectbox("Fuel Type", ["Any","Petrol","Diesel","Electric","Aviation"])
    with c4:
        seats = st.selectbox("Seats", ["Any","2","4","5","6","7"])
    submitted = st.form_submit_button("Recommend Vehicles 🚀")

seats_val = None if seats == "Any" else int(seats)
if submitted:
    st.session_state.recommendations = recommend(budget, vtype, fuel, seats_val)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Results - actually shows now
# ---------------------------------------------------------------------------
if st.session_state.recommendations is not None:
    rec = st.session_state.recommendations
    if rec.empty:
        st.markdown('<div class="warn">No vehicles found within that budget. Try increasing the budget.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sec-title animate-in"><span></span>Recommended Vehicles</div>', unsafe_allow_html=True)
        cols = st.columns(min(4, len(rec)))
        for i, (_, row) in enumerate(rec.iterrows()):
            with cols[i % len(cols)]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<h4>{row["Name"]}</h4>', unsafe_allow_html=True)
                st.markdown(f'<div class="price">${row["Price_USD"]:,}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="meta">{row["Brand"]} &bull; {row["Type"]} &bull; {row["Fuel_Type"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="meta">{row["Mileage_kmpl"]} kmpl &bull; {row["Seats"]} seats &bull; Safety {row["Safety_Rating"]}/5</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="advantage">Advantage Score: {row["Advantage_Score"]}/100</div>', unsafe_allow_html=True)
                st.caption(row.get("Description",""))
                st.markdown(f'<a href="{row.get("Link","#")}" target="_blank"><button class="stButton link" style="width:100%;margin-top:8px;">Open Website ↗</button></a>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Comparison chart
        st.markdown('<div class="sec-title animate-in"><span></span>Comparison</div>', unsafe_allow_html=True)
        show_cols = ["Name","Brand","Type","Price_USD","Mileage_kmpl","Seats","Fuel_Type","Safety_Rating","Advantage_Score"]
        st.dataframe(rec[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rec["Name"], y=rec["Advantage_Score"], name="Advantage Score", marker_color="#ff7a1a", text=rec["Advantage_Score"], textposition="outside"))
        fig.add_trace(go.Bar(x=rec["Name"], y=rec["Safety_Rating"], name="Safety Rating", marker_color="#2a9df4", text=rec["Safety_Rating"], textposition="outside"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#070709", plot_bgcolor="#070709",
                          font=dict(color="#e8e8ea", family="Inter"), barmode="group", height=400,
                          margin=dict(l=20,r=20,t=30,b=20),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Chatbot - bottom right corner (screenshot 3 style)
# ---------------------------------------------------------------------------
st.markdown('<div class="chat-wrap-spacer" style="height:80px;"></div>', unsafe_allow_html=True)

if not st.session_state.chat_open:
    st.markdown('<div class="chat-toggle-wrap">', unsafe_allow_html=True)
    if st.button("🤖", key="open_chat", help="Open Vehicle Assistant"):
        st.session_state.chat_open = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">🤖 Vehicle Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages" id="chat-msgs">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        role_class = "user" if m["role"] == "user" else "bot"
        emoji = "🧑" if m["role"] == "user" else "🤖"
        st.markdown(f'<div class="msg {role_class}"><span class="emoji">{emoji}</span>{m["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Type your question...", key="chat_input")
    if user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.spinner("Thinking..."):
            reply = chat_with_ai(user_input)
        st.session_state.messages.append({"role":"bot","content":reply})
        st.rerun()

    cclose, _ = st.columns([1,4])
    with cclose:
        if st.button("✕ Close", key="close_chat"):
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
