# Budget-Based Vehicle Recommender

A complete ML pipeline: train on **Kaggle** → push to **GitHub** → deploy on **Streamlit Cloud**.

## What it does
1. You enter your **budget** and preferences.
2. The model recommends vehicles you can afford, ranked by an **advantage score** (value/features).
3. Side-by-side **comparison** of recommended vehicles (table + chart).
4. A floating **multi-AI chatbot** (Gemini + OpenAI + Groq with fallback) answers all follow-up questions.
5. Animated dark UI inspired by the screenshots you shared (entrance motion, hover-active cards, play-button hero).

## Folder structure
```
vehicle_recommender/
├── data/vehicles.csv          # Sample vehicle dataset
├── kaggle_train.ipynb         # Kaggle training notebook (upload & run on Kaggle)
├── train_model.py             # Local training script (same logic as notebook)
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── .env.example               # API key template (copy to .env)
├── model/                     # Trained model artifacts (created after training)
└── README.md
```

## Step 1: Train on Kaggle
1. Go to [kaggle.com](https://www.kaggle.com) → Code → New Notebook.
2. Upload `data/vehicles.csv` to the notebook.
3. Copy the contents of `kaggle_train.ipynb` (or run `train_model.py`) into the notebook.
4. Run all cells. It saves `vehicle_model.joblib` and `vehicle_data.csv` to `/kaggle/working/`.
5. Download those two files from Kaggle.

## Step 2: Add to GitHub
1. Create a new repository on GitHub (e.g., `vehicle-recommender`).
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `data/vehicles.csv`
   - `model/vehicle_model.joblib`  (from Kaggle)
   - `model/vehicle_data.csv`      (from Kaggle)
   - `.env.example`
   - `README.md`
3. Do **NOT** upload a real `.env` with your keys. Add `.env` to `.gitignore`.

## Step 3: Deploy on Streamlit Cloud
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your GitHub repo.
2. Set the main file path to `app.py`.
3. In **Advanced settings → Secrets**, add your API keys (toml format):
   ```toml
   GEMINI_API_KEY = "your-gemini-key"
   OPENAI_API_KEY = "your-openai-key"
   GROQ_API_KEY = "your-groq-key"
   ```
4. Click Deploy. Your app will be live at a `share.streamlit.app` URL.

## Local run
```bash
cd vehicle_recommender
cp .env.example .env
# edit .env with your keys
pip install -r requirements.txt
python train_model.py           # creates model/ artifacts locally
streamlit run app.py
```

## API keys (important)
I cannot provide real Gemini / OpenAI / Groq keys. You must obtain them yourself:
- **Gemini:** https://aistudio.google.com/app/apikey
- **OpenAI:** https://platform.openai.com/api-keys
- **Groq:** https://console.groq.com/keys

The chatbot tries them in order: Gemini → OpenAI → Groq, with a friendly fallback if none are configured.
