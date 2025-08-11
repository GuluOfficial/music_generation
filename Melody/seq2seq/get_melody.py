from random import randrange
import torch.nn as nn
import torch
import os
from Melody.seq2seq.model import Encoder, Decoder
from config import root_path

import pickle

from utils.profile import Transfrom

def Load_Vocab(file):
    with open(file, 'rb') as fd:
        _vocab = pickle.load(fd)
    return _vocab
    
def Load_Parameters(file):
    with open(file, 'rb') as fd:
        parameters_dict = pickle.load(fd)
    return parameters_dict

class LyricMelody(nn.Module):
  def __init__(self):
    super(LyricMelody, self).__init__()
    '''利用seq2seq模型获取音符和时长'''
    
    self.sample_type = 'Beam search'
    self.use_cuda = False
    self.SPs = [0.007100, 6.171630]
    self.Aps = [0.032920, 0.956690]
    
    lyric2note_en_vocab_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2note/best/en_vocab.pkl')
    lyric2note_de_vocab_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2note/best/de_vocab.pkl')
    lyric2note_hyper_parameters_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2note/best/parameters_dict.pkl')

    lyric2note_encoder_model_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2note/best/encoder.37.pt')
    lyric2note_decoder_model_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2note/best/decoder.37.pt')
    
    self.lyric2note_encoder, self.lyric2note_decoder, self.lyric2note_en_vocab, self.lyric2note_de_vocab, self.lyric2note_trf = self.get_model(lyric2note_en_vocab_file, lyric2note_de_vocab_file,lyric2note_hyper_parameters_file,  lyric2note_encoder_model_file, lyric2note_decoder_model_file)
    
    lyric2duration_en_vocab_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2duration/en_vocab.pkl')
    lyric2duration_de_vocab_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2duration/de_vocab.pkl')
    lyric2duration_hyper_parameters_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2duration/parameters_dict.pkl')

    lyric2duration_encoder_model_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2duration/encoder.50.pt')
    lyric2duration_decoder_model_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2duration/decoder.50.pt')
    
    self.lyric2duration_encoder, self.lyric2duration_decoder, self.lyric2duration_en_vocab, self.lyric2duration_de_vocab, self.lyric2duration_trf = self.get_model(lyric2duration_en_vocab_file, lyric2duration_de_vocab_file, lyric2duration_hyper_parameters_file, lyric2duration_encoder_model_file, lyric2duration_decoder_model_file)
    
    
    note2duration_en_vocab_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/note2duration/en_vocab.pkl')
    note2duration_de_vocab_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/note2duration/de_vocab.pkl')
    note2duration_hyper_parameters_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/note2duration/parameters_dict.pkl')

    note2duration_encoder_model_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/note2duration/encoder.27.pt')
    note2duration_decoder_model_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/note2duration/decoder.27.pt')
    
    self.note2duration_encoder, self.note2duration_decoder, self.note2duration_en_vocab, self.note2duration_de_vocab, self.note2duration_trf = self.get_model(note2duration_en_vocab_file, note2duration_de_vocab_file, note2duration_hyper_parameters_file, note2duration_encoder_model_file, note2duration_decoder_model_file)
    
    
  def get_model(self, en_vocab_file, de_vocab_file, hyper_parameters_file, encoder_model_file, decoder_model_file):
    
    parameters_dict = Load_Parameters(hyper_parameters_file)
    en_vocab = Load_Vocab(en_vocab_file)
    de_vocab = Load_Vocab(de_vocab_file)
        
    trf = Transfrom(en_vocab)

    en_embedding_dim = parameters_dict['en_embedding_dim']
    de_embedding_dim = parameters_dict['de_embedding_dim']
    hidden_dim = parameters_dict['hidden_dim']
    num_layers = parameters_dict['num_layers']
    bidirectional = parameters_dict['bidirectional']
    use_lstm = parameters_dict['use_lstm']
    use_cuda = False
    dropout_p = 0.1
        
    encoder = Encoder(en_embedding_dim, hidden_dim, en_vocab.n_items, num_layers, dropout_p, bidirectional, use_lstm, use_cuda)
    decoder = Decoder(de_embedding_dim, hidden_dim, de_vocab.n_items, num_layers, dropout_p, bidirectional, use_lstm, use_cuda)
        
    encoder.load_state_dict(torch.load(encoder_model_file, map_location='cpu'))
    decoder.load_state_dict(torch.load(decoder_model_file, map_location='cpu'))
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, en_vocab, de_vocab, trf
  
  def greedy_search(self, decoder, de_vocab, encoder_output, decoder_outputs, decoder_state, max_len):
    ''''''
    pred_char = ''
    pred_sent = []
    while pred_char != '_EOS_':
        log_prob, v_idx = decoder_outputs.data.topk(1)
        pred_char = de_vocab.index2item[v_idx.item()]
        pred_sent.append(pred_char)

        if(len(pred_sent) > max_len): break
                    
        decoder_inputs = torch.LongTensor([v_idx.item()])
        if self.use_cuda: decoder_inputs = decoder_inputs.cuda()
        decoder_outputs, decoder_state = decoder(decoder_inputs, encoder_output, decoder_state)
        
    return pred_sent
  
  def beam_search(self, decoder, de_vocab, encoder_output, decoder_outputs, decoder_state, max_len):
      samples = []
      pred_sent = []
      topk = 5
      log_prob, v_idx = decoder_outputs.data.topk(topk)
      for k in range(topk):
        samples.append([[v_idx[0][k].item()], log_prob[0][k], decoder_state])

      for _ in range(max_len):
          new_samples = []
                    
          for sample in samples:
              v_list, score, decoder_state = sample
            
              if v_list[-1] == de_vocab.item2index['_EOS_']:
                new_samples.append([v_list, score, decoder_state])
                continue
                        
              decoder_inputs = torch.LongTensor([v_list[-1]])

              decoder_outputs, new_states = decoder(decoder_inputs, encoder_output, decoder_state)
              log_prob, v_idx = decoder_outputs.data.topk(topk)
                        
              for k in range(topk):
                  new_v_list = []
                  new_v_list += v_list + [v_idx[0][k].item()]
                  new_samples.append([new_v_list, score + log_prob[0][k], new_states])

          new_samples = sorted(new_samples, key = lambda sample: sample[1], reverse=True)
          samples = new_samples[:topk]

      best_score = -(1e8)
      best_idx = -1
      best_states = None
      for i, sample in enumerate(samples):
          v_list, score, states = sample
          if score.item() > best_score:
              best_score = score
              best_idx = i
              best_states = states

      v_list, score, states = samples[best_idx]
      for v_idx in v_list:
          pred_sent.append(de_vocab.index2item[v_idx])
          
      return pred_sent
  
  def decode_note(self, content):
    '''由歌词预测音符'''
    
    en_seq, en_seq_len = self.lyric2note_trf.trans_input(content)

    en_seq = torch.LongTensor(en_seq)
    encoder_input = en_seq
    encoder_output, encoder_state = self.lyric2note_encoder(encoder_input, en_seq_len)
    
    decoder_state = self.lyric2note_decoder.init_state(encoder_state)

    # Start decoding
    decoder_inputs = torch.LongTensor([self.lyric2note_de_vocab.item2index['_START_']])
    
    pred_char = ''
    pred_sent = []
    if self.use_cuda: decoder_inputs = decoder_inputs.cuda()
    decoder_outputs, decoder_state = self.lyric2note_decoder(decoder_inputs, encoder_output, decoder_state)

    max_len = len(content.split())
    
    if self.sample_type == 'Greedy':
      # Greedy search
      pred_sent = self.greedy_search(self.lyric2note_decoder, self.lyric2note_de_vocab, encoder_output, decoder_outputs, decoder_state, max_len) # decoder, de_vocab, encoder_output, max_len
    else:
      # Beam search
      pred_sent = self.beam_search(self.lyric2note_decoder, self.lyric2note_de_vocab, encoder_output, decoder_outputs, decoder_state, max_len)
            
    pred_list = []
            
    scoring_stop = False
    for i in range(max_len):
        if not scoring_stop:
          if pred_sent[i] == '_EOS_':
            scoring_stop = True
          if pred_sent[i] != '_EOS_':
            pred_list.append(pred_sent[i])
            
    
    return pred_list
  
  def decode_note2duration(self, notes):
    '''由音符序列预测时长问题'''
    en_seq, en_seq_len = self.note2duration_trf.trans_input(notes)

    en_seq = torch.LongTensor(en_seq)
    encoder_input = en_seq
    encoder_output, encoder_state = self.note2duration_encoder(encoder_input, en_seq_len)
    
    decoder_state = self.note2duration_decoder.init_state(encoder_state)

    # Start decoding
    decoder_inputs = torch.LongTensor([self.note2duration_de_vocab.item2index['_START_']])
    
    pred_sent = []
    if self.use_cuda: decoder_inputs = decoder_inputs.cuda()
    decoder_outputs, decoder_state = self.note2duration_decoder(decoder_inputs, encoder_output, decoder_state)

    max_len = len(notes.split())
    
    if self.sample_type == 'Greedy':
      # Greedy search
      pred_sent = self.greedy_search(self.note2duration_decoder, self.note2duration_de_vocab, encoder_output, decoder_outputs, decoder_state, max_len) # decoder, de_vocab, encoder_output, max_len
    else:
      # Beam search
      pred_sent = self.beam_search(self.note2duration_decoder, self.note2duration_de_vocab, encoder_output, decoder_outputs, decoder_state, max_len)
            
    pred_list = []
            
    scoring_stop = False
    for i in range(max_len):
        if not scoring_stop:
          if pred_sent[i] == '_EOS_':
            scoring_stop = True
          if pred_sent[i] != '_EOS_':
            pred_list.append(pred_sent[i])
            
    return pred_list
    
    
  
  def decode_duration(self, content):
    '''由歌词预测时长问题'''
    
    en_seq, en_seq_len = self.lyric2duration_trf.trans_input(content)

    en_seq = torch.LongTensor(en_seq)
    encoder_input = en_seq
    encoder_output, encoder_state = self.lyric2duration_encoder(encoder_input, en_seq_len)
    
    decoder_state = self.lyric2duration_decoder.init_state(encoder_state)

    # Start decoding
    decoder_inputs = torch.LongTensor([self.lyric2duration_de_vocab.item2index['_START_']])
    
    pred_sent = []
    if self.use_cuda: decoder_inputs = decoder_inputs.cuda()
    decoder_outputs, decoder_state = self.lyric2duration_decoder(decoder_inputs, encoder_output, decoder_state)

    max_len = len(content.split())
    
    if self.sample_type == 'Greedy':
      # Greedy search
      pred_sent = self.greedy_search(self.lyric2duration_decoder, self.lyric2duration_de_vocab, encoder_output, decoder_outputs, decoder_state, max_len) # decoder, de_vocab, encoder_output, max_len
    else:
      # Beam search
      pred_sent = self.beam_search(self.lyric2duration_decoder, self.lyric2duration_de_vocab, encoder_output, decoder_outputs, decoder_state, max_len)
            
    pred_list = []
            
    scoring_stop = False
    for i in range(max_len):
        if not scoring_stop:
          if pred_sent[i] == '_EOS_':
            scoring_stop = True
          if pred_sent[i] != '_EOS_':
            pred_list.append(pred_sent[i])
            
    return pred_list
  
  def get_melody(self, lyrics):
    '''歌词生成音符和时长序列'''
    content = lyrics.replace(',', '')
    content = lyrics.replace('，', '')
    content = ' '.join(content).strip()
    
    notes = self.decode_note(content)
    # durations = self.decode_duration(content)
    str_notes = ' '.join(notes)
    durations = self.decode_note2duration(str_notes)
    
    #补齐序列--模型漏洞
    max_len = len(content.split())
    
    if len(notes) != max_len:
      last = notes[-1]
      while len(notes) < max_len:
        notes.append(last)
    
    if len(durations) != max_len:
      mean = 0.4202
      while len(durations) < max_len:
        durations.append(mean)
        
    #在歌词中加入特殊停顿或者吸气
    #在音符序列对应的位置加入rest
    #在时长序列对应的位置加入对应的时长
    
    if ',' in lyrics or '，' in lyrics:
      '''如果歌词中有逗号，逗号替换为SP，或者AP,且在歌词末尾添加SP或者AP
      有可能有多个逗号
      '''
      new_lyrics = list(lyrics)
      new_notes = []
      new_durations = []
      re_lyrics = []
      i = 0
      
      for lyric in new_lyrics:
        if (',' not in lyric) and ('，' not in lyric):
          re_lyrics.append(lyric)
          new_notes.append(notes[i])
          new_durations.append(durations[i])
          i = i + 1
          
        else:
          new_notes.append('rest')
          new_notes.append('rest')
          re_lyrics.append('SP')
          re_lyrics.append('AP')
          new_durations.append('0.2102')
          new_durations.append('0.3024')
        
      return re_lyrics, new_notes, new_durations
    else:
      '''
      如果歌词中没有逗号：
      1、直接在末尾添加SP或者AP
      2、在三个序列中随机插入SP和AP,在末尾加入AP
      '''
      lyrics = list(lyrics)
      index =  randrange(0, len(lyrics) - 1)
      
      if len(lyrics) > 6:
        
        while index == 0 or index == len(lyrics) - 1:
          index =  randrange(0, len(lyrics) - 1)
        
        if index < len(lyrics) - 1:
          lyrics.insert(index, 'SP')
          notes.insert(index, 'rest')
          durations.insert(index, '0.1104')
          
          lyrics.insert(index + 1, 'AP')
          notes.insert(index + 1, 'rest')
          durations.insert(index + 1, '0.3104')
        else:
          lyrics.insert(index, 'AP')
          notes.insert(index, 'rest')
          durations.insert(index, '0.3104')
          
      
      lyrics = ''.join(lyrics) + 'AP'
      notes.append('rest')
      durations.append('0.3104')
          
      
      return ''.join(lyrics), notes, durations
    
    


      
      

      
    
    
    
    
    
    
            
    

    
    
    