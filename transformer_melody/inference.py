
import numpy as np
import os
from transformer_melody.Lyric2Duration import Lyric2Duration
from transformer_melody.beam_decoder import beam_search
from transformer_melody.model import make_model
from config import root_path
import torch
import pickle

from transformer_melody.modules import batch_greedy_decode

def load_pkl(path):
    with open(path, 'rb') as f:
        dictionary = pickle.load(f)
    return dictionary


def getmodel(device):
  '''获取lyric2note和lyric2duration的模型'''
  lyrics_dictionary_file = os.path.join(root_path, 'transformer_melody/saved_dictionary/lyrics_dictionary.pkl')
  notes_dictionary_file = os.path.join(root_path, 'transformer_melody/saved_dictionary/notes_dictionary.pkl')

  lyric_dictionary = load_pkl(lyrics_dictionary_file)
  note_dictionary = load_pkl(notes_dictionary_file)

  src_vocab_size = lyric_dictionary.vocabulary_size
  tgt_vocab_size = note_dictionary.vocabulary_size
  
  n_layers = 6
  d_model = 512
  d_ff = 2048
  n_heads = 8
  dropout = 0.1
  # 初始化模型
  lyric2note = make_model(src_vocab_size, tgt_vocab_size, n_layers, d_model, d_ff, n_heads, dropout)
    
  lyric2note_path = os.path.join(root_path, 'transformer_melody/saved_models/transformer_flow/lyric2note_0530.pt')
  state_dict = torch.load(lyric2note_path, map_location=device)
  lyric2note.load_state_dict(state_dict)
  
  lyric2duration_path = os.path.join(root_path, 'transformer_melody/saved_models/transformer_flow/lyric2duration_0530.pt')

  lyric_dictionary = load_pkl(lyrics_dictionary_file)
  input_dim = lyric_dictionary.vocabulary_size

  hidden_dim = 256
  enc_layers = 6
  enc_heads = 8
  enc_pf_dim = 512
  enc_dropout = 0.1
  src_pad_idx = 0
  trg_pad_idx = 0
  use_sdp = True

  lyric2duration = Lyric2Duration(input_dim, hidden_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 use_sdp,
                 device).to(device)

  state_dict = torch.load(lyric2duration_path, map_location=device)
  lyric2duration.load_state_dict(state_dict)
  
  lyric2note.to(device)
  lyric2duration.to(device)

  return lyric2note, lyric2duration, lyric_dictionary, note_dictionary

def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len >= n:
        return array[: n]
    extra = n - current_len
    return array + ([0] * extra)
  

def genarate_melody(lyric2note, lyric2duration, lyric_dictionary, note_dictionary, lyrics, device, index, previous_note):
  '''
  依据tansformer的结构推理音符序列和时长序列
  lyrics在接口里是一句歌词
  '''
  
  BOS = lyric_dictionary.indexer('<BOS>')
  EOS = lyric_dictionary.indexer('<EOS>')
  
  if index == 0:
    
    src_tokens = [[BOS] + lyric_dictionary.encode(lyrics) + [EOS]]
    src = torch.LongTensor(np.array(src_tokens)).to(device)
    src_mask = (src != 0).unsqueeze(-2)
    notes = infer_note(lyric2note, src, src_mask, len(lyrics), note_dictionary, use_beam=True, device=device, first=True)
    
  else:
      
    src_tokens = [BOS] + lyric_dictionary.encode(lyrics) + [EOS]
    previous_note = previous_note[:-1] # 此处是去掉上一句末尾的rest特殊符号，不带入模型参与预测
    previous_ids = [BOS] + note_dictionary.encode(previous_note) + [EOS]
    previous_ids = rpad(previous_ids, n=60)
    src_tokens = rpad(src_tokens, n=60)

    src = torch.LongTensor(np.array([src_tokens])).to(device)
    src_mask = (src != 0).unsqueeze(-2).to(device)

    previous_input = torch.LongTensor(np.array([previous_ids])).to(device)
    notes = infer_note(lyric2note, src, src_mask, len(lyrics), note_dictionary, use_beam=True, device=device, first=True, previous_input=previous_input) 

  notes = notes[:len(lyrics)]
  notes = handle_notes(notes, target_len=len(lyrics))
  src_tokens = [[BOS] + lyric_dictionary.encode(lyrics) + [EOS]]
  src = torch.LongTensor(np.array(src_tokens)).to(device)
  durations = lyric2duration.infer_duration(src)
  durations = durations.detach().cpu().flatten().numpy().tolist()
  durations = [round(float(d * 0.01), 4) for d in durations]
  durations = durations[1:len(lyrics) + 1]
  
  #时长需要特殊处理一下，有的是有点短
  durations = handle_durations(durations)
  
  #特殊处理一下
  notes.append('rest')
  durations.append(0.6)
  lyrics += 'AP'
  
  
  return lyrics, notes, durations

def handle_durations(durations):
    '''特殊处理durations,当token对应的时长太短，就暂时定为0.3201，如果token对应的时长太长的，暂时就定为0.3405'''
    
    duras = []
    for d in durations:
        if d <= 0.15:
            d = 0.3201
        elif d >= 2:
            d = 0.4305
        duras.append(d)
    
    return duras
  

def handle_notes(notes, target_len):
    '''notes序列生成的长度有可能不够，需要特殊处理一下'''
    news = []
    for n in notes:
        if n == '<EOS>':
            continue
        news.append(n)
    
    while len(news) < target_len:
        news.append(news[-1])
    
    if len(news) > target_len:
        news = news[:target_len]
    
    return news
 
def infer_note(model, src, src_mask, max_len, note_dictionary, use_beam=True, device=None, first=True, previous_input=None): 
  sp_chn = note_dictionary
  bos_idx = note_dictionary.indexer('<BOS>')
  eos_idx = note_dictionary.indexer('<EOS>')
  padding_idx = 0
  beam_size = 3
  
  with torch.no_grad():
      model.eval()

      if use_beam:
          decode_result, _ = beam_search(model, src, src_mask, max_len,
                                           padding_idx, bos_idx, eos_idx,
                                           beam_size, device, first, previous_input)
          decode_result = [h[0] for h in decode_result]
      else:
          decode_result = batch_greedy_decode(model, src, src_mask, max_len=max_len)

       
      translation = [sp_chn.decode_ids(_s) for _s in decode_result]
      return translation[0]
  
  
  
  