from fastapi import FastAPI, UploadFile, File, HTTPException
from uuid import uuid4
from pydantic import BaseModel
import requests
import json
import io
import sys
import torch
import numpy as np
sys.path.append('./app')
from models import Quesgen, SeqSearch

sys.path.append('')

app = FastAPI()
raw_model = 'cointegrated/rut5-base-multitask'
seqsearch = SeqSearch('weights/search.pth', raw_model)
quesgen = Quesgen('weights/firstmodel.pth', raw_model)

class TextInput(BaseModel):
    url: str


@app.post("/")
async def Quetion_generation(count: int, input: TextInput):
    url = input.url
    try:
        response = requests.get(url)
        r = response.json()
        if response.status_code == 200:
            data = np.array(r)
        else:
            HTTPException(status_code=404, detail="Failed to download file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    choice = list(np.random.choice(len(data), count, replace=False))
    data = data[choice]
    res = dict({'questions': []})
    for par in data:
        paragraph = ' '.join(par['content'])
        context = seqsearch.generate(paragraph, max_length=200)
        question = quesgen.generate("", context, max_length=128)
        res['questions'].append({
            'question': question,
            'title': par['title']
        })

    return res
