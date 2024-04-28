from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from transformers import pipeline
import torch


class Quesgen:

    def __init__(self, weights_path, raw_model, device='cpu'):
        self.model = T5ForConditionalGeneration.from_pretrained(raw_model).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(raw_model)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    def generate(self, answer, context, **generator_args):
        input_ids = self.tokenizer.encode(f'answer: , context: {context}', return_tensors="pt").to(self.model.device)
        res = self.model.generate(input_ids, **generator_args)
        output = self.tokenizer.decode(res[0], skip_special_tokens=True)
        return output

class SeqSearch:

    def __init__(self, weights_path, raw_model, device='cpu'):
        self.model = T5ForConditionalGeneration.from_pretrained(raw_model).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(raw_model)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    def generate(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
        res = self.model.generate(input_ids, **generator_args)
        output = self.tokenizer.decode(res[0], skip_special_tokens=True)
        return output


class QuesAns:

    def init(self):
        self.pipe = pipeline("question-answering", model="timpal0l/mdeberta-v3-base-squad2")

    def generate(self, question, context):
        return self.pipe(question=question, context=context)['answer']