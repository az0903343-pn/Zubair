# Vehicle Budget Recommender

Train on **Kaggle** → Push to **GitHub** → Deploy on **Streamlit Cloud**.

## Kya karta hai ye app
1. Aap **budget** dalte hain → model aapke budget ke hisaab se best vehicles recommend karta hai (**Advantage Score** ke hisaab se).
2. Side-by-side **comparison** table + chart dikhata hai.
3. Har vehicle ke saath **direct website link** button hota hai.
4. **Bottom-right corner** mein **ChatGPT jaisa chatbot** (Gemini → OpenAI → Groq fallback) hota hai jo sawaalon ke jawab deta hai.
5. Animated dark UI: intro/demo screen, "Vehicle Finder" name, category cards (cycle se helicopter tak).

## Folder structure
```
done/
├── app.py                      # Streamlit application (main file)
├── train_model.py              # Train KNN recommender locally
├── generate_data.py            # 15,000 vehicles ka synthetic dataset banaye
├── kaggle_train.ipynb          # Kaggle notebook (upload & run on Kaggle)
├── requirements.txt            # Python dependencies
├── .env.example                # API key template (copy to .env)
├── .gitignore                  # Na commit karein real keys ko
└── data/
    └── vehicles.csv            # Sample data (shuruat ke liye)
```

## Step 1: Data taiyar karein (15,000 vehicles)
Agar aap chahte hain ke model mein **15,000 vehicles (cycle se helicopter tak)** hon, to ye command run karein:

```bash
python generate_data.py
```

Yeh `data/vehicles_15000.csv` banayega. Isse Kaggle notebook mein upload karke train karein.

## Step 2: Train on Kaggle
1. [kaggle.com](https://www.kaggle.com) → Code → New Notebook.
2. `data/vehicles_15000.csv` (ya `vehicles.csv`) upload karein.
3. `kaggle_train.ipynb` ka code copy karke run karein.
4. `/kaggle/working/model/` se ye do files download karein:
   - `vehicle_model.joblib`
   - `vehicle_data.csv`

## Step 3: GitHub par upload karein
Naya GitHub repository banayein aur ye files upload karein:
- `app.py`
- `requirements.txt`
- `data/vehicles.csv` (sample)
- `model/vehicle_model.joblib` (Kaggle se download ki hui)
- `model/vehicle_data.csv` (Kaggle se download ki hui)
- `README.md`
- `.env.example`
- `.gitignore`

**Real `.env` file GitHub par mat upload karein.** `.gitignore` mein `.env` pehle se hai.

## Step 4: Streamlit Cloud par deploy karein
1. [streamlit.io/cloud](https://streamlit.io/cloud) par GitHub repo connect karein.
2. Main file path: `app.py`.
3. **Advanced settings → Secrets** mein yein keys add karein (TOML format):
   ```toml
   GEMINI_API_KEY = "aap-ka-gemini-key"
   OPENAI_API_KEY = "aap-ka-openai-key"
   GROQ_API_KEY = "aap-ka-groq-key"
   ```
4. Deploy click karein. App live ho jayegi.

## API keys (aap khud layenge — main real keys nahi de sakta)
- **Gemini:** https://aistudio.google.com/app/apikey
- **OpenAI:** https://platform.openai.com/api-keys
- **Groq:** https://console.groq.com/keys

Chatbot inmein se kisi ek key ke bina nahi chalega. Agar koi key nahi hai toh chatbot error dega, lekin **vehicle recommender abhi bhi kaam karega**.

## Local run
```bash
cd done
pip install -r requirements.txt
python train_model.py          # model/ folder banata hai locally
streamlit run app.py
```

## Internet se 15,000 asli vehicles ka data
Main asli internet se data scrape nahi kar sakta. Agar aap chahte hain ke model mein **asli vehicle data** ho, to:
- Kaggle par "vehicle dataset" search karke CSV download karein.
- Ya kisi car/bike website ka API use karein.
- Data ko `data/vehicles.csv` mein save karke `train_model.py` run karein.

Har vehicle ke saath `Link` column hota hai jo direct website kholta hai (app mein "Open Website ↗" button).
