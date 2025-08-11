import torch, os, json, random, logging
import numpy as np
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from simplet5 import SimpleT5
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class MengziSimpleT5_infer(SimpleT5):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda")

    def load_my_model(self, use_gpu: bool = True):
        self.tokenizer = T5Tokenizer.from_pretrained('./Lyric_generator_t5/checkpoint')
        self.model = T5ForConditionalGeneration.from_pretrained('./Lyric_generator_t5/checkpoint')

def setup_lyric():
    model = MengziSimpleT5_infer()
    model.load_my_model()
    model.model = model.model.to('cuda')  
    return model


def generate_lyric(lyric_title, model):
    TITLE_PROMPT = "关键词："
    LENGTH_PROMPT = "长度："
    EOS_TOKEN = ' '
    in_request = TITLE_PROMPT + lyric_title + EOS_TOKEN + LENGTH_PROMPT + "15"
    out = model.predict(in_request, max_length=512, num_return_sequences=1, top_k=10)

    return out[0]