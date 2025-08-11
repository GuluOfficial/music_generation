import torch, os, json, random, logging
import numpy as np
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from transformers import  BertTokenizerFast, GPT2LMHeadModel

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def setup_lyric():
    special_tokens = {
    'additional_special_tokens': ["<Folk 民谣>", "<Soundtrack 原声>", "<Ancientry 古风>", "<Rap/Hip Hop & Dance 舞曲>", "<Children Music 儿童音乐>", "<Pop 流行>", "<Rock 摇滚 & Metal 金属>", "<结束>"]
    }
    tokenizer = BertTokenizerFast(vocab_file='./Lyric_generator/checkpoint/vocab.txt', sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()
    model = GPT2LMHeadModel.from_pretrained('./Lyric_generator/checkpoint/')
    model.resize_token_embeddings(len(vocab))
    return tokenizer, model


def generate_lyric(lyric_title, tokenizer, model, lyric_style="<Pop 流行>"):
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    text_ids = tokenizer.encode(lyric_title, add_special_tokens=False)
    style_ids = tokenizer.encode(lyric_style)

    input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
    input_ids.extend(text_ids)
    input_ids.append(tokenizer.sep_token_id)
    
    input_ids.extend([style_ids[1]])
    input_ids.append(tokenizer.sep_token_id)

    input_ids = torch.tensor(input_ids).long().to(device)
    input_ids = input_ids.unsqueeze(0)

    beam_outputs = model.generate(
        input_ids, 
        max_length=90, 
        num_beams=1,  
        no_repeat_ngram_size=2,
        num_return_sequences=1
    )
    for i in beam_outputs:
        fin_out = tokenizer.decode(i,skip_special_tokens=False).split('<结束>')[0].split('[CLS]')[1]
    
    return "，".join(fin_out.split('[SEP]')[2:-1]).replace(' ','')