from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from joblib import load
from fun import remove_num, remove_emoj, remove_other, remove_punc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models load karna
model = load('spam_model.pkl')
vectorizer = load('vectorizer.pkl')

class Text(BaseModel):
    text: str = Field(
        ...,
        max_length=1000,
        min_length=1,
        description="Text to be classified"
    )

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