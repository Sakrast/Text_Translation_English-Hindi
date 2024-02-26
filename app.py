"""





Please download the saved model through the given link then run <python app.py>

https://drive.google.com/uc?export=download&id=1xvkw8MIYKEA2b2KyJH1WFKbtmNreQRsq












"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
import uvicorn
import re
import string
from transformers import MarianMTModel, MarianTokenizer


app = FastAPI()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


model_dir = "model"  
tokeniser_dir = "tokenizer" 

model = MarianMTModel.from_pretrained(model_dir, from_tf=True) 
tokeniser = MarianTokenizer.from_pretrained(tokeniser_dir)


class InputData(BaseModel):
    text: str = "                        Please enter the text you want to translate                               "


def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


@app.post('/predict')
async def predict(data: InputData):
    text = clean_text(data.text)
    inputs = tokeniser.prepare_seq2seq_batch(src_texts=[text], return_tensors="pt")
    translate_ids = model.generate(**inputs)
    translated_text = tokeniser.decode(translate_ids[0], skip_special_tokens=True)
    return {'translate': translated_text}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
