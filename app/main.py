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
from models import Quesgen, SeqSearch, QuesAns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('')

app = FastAPI()
raw_model = 'cointegrated/rut5-base-multitask'
seqsearch = SeqSearch('weights/search.pth', raw_model)
quesgen = Quesgen('weights/firstmodel.pth', raw_model)
quesans = QuesAns()

class TextInput(BaseModel):
    url: str


@app.post("/")
async def Quetion_generation(count: int, input: TextInput, tem: str|None=None):
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
    if tem != None:
        for i, el in enumerate(data):
            if el['title'] == tem:
                choice = list([i])
    else:
        choice = list(np.random.choice(len(data), count, replace=False))
    data = data[choice]
    res = dict({'questions': []})
    for par in data:
        paragraph = ' '.join(par['content'])
        context = seqsearch.generate(paragraph, max_length=200)
        question = quesgen.generate("", context, max_length=128)
        res['questions'].append({
            'question': question,
            'title': par['title'],
            'context': context
        })

    return res

@app.post("/check")
async def Quetion_check(input: TextInput):
    url = input.url
    try:
        response = requests.get(url)
        r = response.json()
        if response.status_code == 200:
            data = np.array(r['questions'])
        else:
            HTTPException(status_code=404, detail="Failed to download file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    res = dict({'questions': []})
    for par in res:
        answerm = quesans.generate(par['question'], par['context'])
        answeru = par['answer']
        documents = [answerm, answeru]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(documents)
        similarity = cosine_similarity(matrix)
        res['questions'].append(similarity[0][1])

    return res