from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from utils.fun import remove_num, remove_emoj, remove_other, remove_punc
from schemas.payload import Text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models load karna
model = load('models/spam_model.pkl')
vectorizer = load('models/vectorizer.pkl')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/predict')
def predict(payload: Text):
    text = payload.text
    text = remove_num(text)
    text = remove_emoj(text)
    text = remove_other(text)
    text = remove_punc(text)
    
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    
    # .item() ya int()/str() se numpy type ko standard Python type me convert karein
    pred = prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0]
    
    return {"prediction": pred}