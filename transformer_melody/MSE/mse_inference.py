import os
import pickle
import numpy as np
import torch
from config import root_path
from transformer_melody.MSE.model import make_model_mse
from transformer_melody.beam_decoder import beam_search
from transformer_melody.model import make_model
from transformer_melody.modules import batch_greedy_decode

def load_pkl(path):
    with open(path, 'rb') as f:
        dictionary = pickle.load(f)
    return dictionary

def load_melody_model_mse(device):
  
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
  lyric2duration = make_model_mse(src_vocab_size, n_layers, d_model, d_ff, n_heads, dropout)
    
  model_path = os.path.join(root_path, 'transformer_melody/saved_models/transformer_linear/lyric2duration_mse30.pt')
  state_dict = torch.load(model_path)
  lyric2duration.load_state_dict(state_dict)
  
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
  
  lyric2note.to(device)
  lyric2duration.to(device)
  
  return lyric2note, lyric2duration, lyric_dictionary, note_dictionary


def get_duration(src, src_mask, model):
  encoder_outputs = model.encode(src, src_mask)

  outputs = model.decoder(encoder_outputs)

  outputs = model.generator(outputs)

  return outputs

def format_duration(outs, mask):
    outs = (torch.exp(outs) - 1) * mask
    outs = torch.round(outs)
    outs = outs.data.cpu().flatten().numpy().tolist()
    value = mask.sum().data
    outs = outs[1:value-1]
    outs = [round(d * 0.01, 4) for d in outs]
    return outs
  

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
  


def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len >= n:
        return array[: n]
    extra = n - current_len
    return array + ([0] * extra)
  
def generate_melody_mse(lyric2note, lyric2duration, lyric_dictionary, note_dictionary, lyric, device, index, previous):
  '''此处lyrics是字符串'''
  
  BOS = lyric_dictionary.indexer('<BOS>')
  EOS = lyric_dictionary.indexer('<EOS>')

  
  if index == 0:

    src_tokens = [[BOS] + lyric_dictionary.encode(lyric) + [EOS]]
    src = torch.LongTensor(np.array(src_tokens)).to(device)
    src_mask = (src != 0).unsqueeze(-2)
    duras = get_duration(src, src_mask, lyric2duration)
    notes = infer_note(lyric2note, src, src_mask, len(lyric), note_dictionary, use_beam=True, device=device, first=True)
  else:
    # 当前歌词对应音符的生成的memory模块和上句歌词生成的音符信息链接一起，
    src_tokens = [BOS] + lyric_dictionary.encode(lyric) + [EOS]
            
    # TODO previous的维度要不要扩充当前的输入长度一致,padding到一个长度，60
    previous_ids = [BOS] + note_dictionary.encode(previous) + [EOS]
    previous_ids = rpad(previous_ids, n=60)
    src_tokens = rpad(src_tokens, n=60)

    src = torch.LongTensor(np.array([src_tokens])).to(device)
    previous_input = torch.LongTensor(np.array([previous_ids])).to(device)
    src_mask = (src != 0).unsqueeze(-2).to(device)
    # 此处时长是代入里面生成还是外面生成，合理的是里面生成即可
    duras = get_duration(src, src_mask, lyric2duration)

    notes = infer_note(lyric2note, src, src_mask, len(lyric), note_dictionary, use_beam=True, device=device, first=True, previous_input=previous_input)
      
  durations = format_duration(duras, src_mask)
  notes = handle_notes(notes, len(lyric))
            
  notes.append('rest')
  durations.append(0.6)
  lyric += 'AP'
  
  return lyric, notes, durations

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
  
  
  