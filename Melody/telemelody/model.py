import os
from config import root_path
from fairseq.models.transformer import TransformerModel
import re
import miditoolkit
from Melody.telemelody.utils import adapt, adapt_e, enc_tpo, enc_ts, enc_vel, encoding_to_midi, get_notes
import torch.nn as nn 

_PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

##dim 减三和弦，小三度音程的上方再加上一个小三度，就可构成减三和弦,指三个音的构成是按照一个半音，再加一个全音构成。
#和弦基础

_CHORD_KIND_PITCHES = {
    '': [0, 4, 7],
    'm': [0, 3, 7],
    '+': [0, 4, 8],
    'dim': [0, 3, 6],
    '7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'm7b5': [0, 3, 6, 10],
}
NO_CHORD = 'N.C.'

pitch_dict = dict()
for idx, name in enumerate(_PITCH_CLASS_NAMES):
    pitch_dict[name] = idx

#25*129=3225 控制时长 
Duration_vocab = dict([(float(x/100), 129+i) for i, x in enumerate(list(range(25, 3325, 25)))])
MAX_DUR = int(max(Duration_vocab.values()))


def clean(word):
    word = re.sub('[ \xa0]+', '', word)
    word = re.sub('[,，] *', ',', word)
    word = re.sub('[。！？?] *', ".", word)
    word = re.sub('.{6} *', ".", word)
    word = re.sub('…+ *', ".", word)
    return word


SEP = '[sep]'
WORD = '[WORD]'

C2 = 36
C3 = 48
min_oct = 5
max_oct = 6
pos_resolution = 4  # per beat (quarter note)
bar_max = 256
velocity_quant = 4
tempo_quant = 12  # 2 ** (1 / 12)
min_tempo = 16
max_tempo = 256
duration_max = 4  # 4 * beat
max_ts_denominator = 6  # x/1 x/2 x/4 ... x/64
max_notes_per_bar = 2  # 1/64 ... 128/64
beat_note_factor = 4  # In MIDI format a note is always 4 beats
deduplicate = True
filter_symbolic = False
filter_symbolic_ppl = 16
trunc_pos = 2 ** 16  # approx 30 minutes (1024 measures)
ts_filter = False

min_pitch = 48
max_pitch = 72


class Lyric2Melody(nn.Module):
  def __init__(self):
    super(Lyric2Melody, self).__init__()
    lyric2beat_prefix = os.path.join(root_path, 'Melody/checkpoints/telemelody')
    trend2note_prefix = os.path.join(root_path, 'Melody/checkpoints/telemelody/')
    
    self.lyric2beats = TransformerModel.from_pretrained(
        lyric2beat_prefix,
        checkpoint_file='lyric2rhythm_best.pt',
        # data_name_or_path=lyric_dict,
    )
    
    self.trend2notes = TransformerModel.from_pretrained(
        trend2note_prefix,
        checkpoint_file='template2melody_best.pt',
        # data_name_or_path=beat_dict
    )

  
    
  def generate_melody(self, sents, bar_chords='C: G: C: E: G: C: F: C: F: A:'):
      '''
      歌词输入，和弦目前预定义
      lyrics：'明 月 几 时 有 [sep] 把 酒 问 青 天 [sep]'
      chord:'C:m E F C'
      '''
      sents = sents.strip()
      syllables = sents
      tmp = bar_chords.split()
      bar_chords = []
      
      for item in tmp:
          if len(bar_chords) >= 2 and item == bar_chords[-1] and item == bar_chords[-2]:
                continue
          bar_chords.append(item)
          
      tmp = []
      cur_period = False
      cur_length = 0
      align_idxs = []
      
      for item in syllables.split():
          if item == SEP:
              if cur_length <= 0:
                  continue
              else:
                  cur_length = 0
                  if cur_period:
                      tmp.append('.')
                  else:
                      tmp.append(',')
                  cur_period = not cur_period
          else:
              if item[0] != '@':
                  align_idxs.append(len([i for i in tmp if i not in [',', '.']]))
              tmp.append(item)
              cur_length += 1

      if tmp[-1] != '.':
          tmp[-1] = '.'

      word_num = len([i for i in tmp if i not in [',', '.']])

            
      beats = self.lyric2beats.translate(syllables,
                                          sampling=True,
                                          sampling_topk=2,
                                          temperature=0.5,
                                          beam=1,
                                          verbose=True,
                                          max_len_a=1,
                                          max_len_b=0,
                                          min_len=len(syllables.split()))

      beats_words = len([i for i in beats.split() if i not in ['[sep]']])

      beats_label = []
      cur_beats = []
      for item in beats.split():
          if item not in ['[sep]', WORD]:
              try:
                  cur_label = int(item)
              except BaseException as e:
                  if len(beats_label):
                      cur_label = beats_label[-1]
                  else:
                      cur_label = 0
              beats_label.append([cur_label])
          if len(beats_label) == word_num:
              break

      cur_idx = 0
      pattern = []
      cur_sent = []
      cur_sep = []
      word_idx = 0
      for word in tmp:
          if word not in [',', '.']:
            cur_sep.extend(beats_label[word_idx])
            cur_idx += len(beats_label[word_idx])
            word_idx += 1
          elif word == ',':
              if len(cur_sep):
                  cur_sent.append(cur_sep)
                  cur_sep = []
          elif word == '.':
              if len(cur_sep):
                  cur_sent.append(cur_sep)
                  cur_sep = []
              if len(cur_sent):
                  pattern.append(cur_sent)
                  cur_sent = []

      pattern = adapt(pattern)
      mode = 'MAJ'

      bar_int = len(bar_chords)

      words = [mode]
      cur_bar = 0
      chords = []

      for sent_idx, sent in enumerate(pattern):
          for sect_idx, section in enumerate(sent):
              next_bar = False
              cur_chord = bar_chords[cur_bar % bar_int]
              print(cur_chord, end=' ')
              for idx, beat in enumerate(section):
                  if next_bar:
                    cur_bar += 1
                    cur_chord = bar_chords[cur_bar % bar_int]
                    print(cur_chord, end=' ')
                  next_bar = False
                  words.append(f'Chord_{cur_chord}')
                  chords.append(cur_chord)
                  if idx != len(section) - 1:
                      words.append('NOT')
                      if section[idx] > section[idx + 1]:
                          next_bar = True
                  elif sect_idx == len(sent) - 1:
                        words.append('AUT')
                  else:
                        words.append('HALF')
                  words.append(f'BEAT_{beat}')

              cur_bar += 1
      trend = ' '.join(words)

      def fix(items):
          tmp = []
          target_tokens = ['Bar', 'Pos', 'Pitch', 'Dur']
          i = 0
          for item in items:
              if item.split('_')[0] == target_tokens[i]:
                  tmp.append(item)
                  i = (i + 1) % len(target_tokens)
          return tmp
      notes = self.trend2notes.translate(trend,
                                          sampling=True,
                                          sampling_topk=10,
                                          temperature=0.5,
                                          max_len_a=4 / 3,
                                          max_len_b=-4 / 3,
                                          min_len=(len(trend.split()) - 1) * 4 // 3,
                                          verbose=True,
                                          beam=1,
                                          )

      enc = fix(notes.split())

      e = list(map(lambda x: int(''.join(filter(str.isdigit, x))), enc))
      print(len(enc) // 4)
      e = [(e[i], e[i + 1], 0, e[i + 2], e[i + 3], enc_vel(127),
                  enc_ts((4, 4)), enc_tpo(80.0)) for i in range(0, len(e) // 4 * 4, 4)]

      min_bar = min([i[0] for i in e])
      e = [tuple(k - min_bar if j == 0 else k for j, k in enumerate(i)) for i in e]
      e.sort()
      e = e[:word_num]
      offset = 0
      e = [tuple(i) for i in e]
      e = adapt_e(e, align_idxs)
      print(e)
      note_chords = []
      for chord, note in zip(chords, e):
          cur_idx = note[0] * 2
          if note[1] >= pos_resolution * 2:
            cur_idx += 1
          if len(note_chords) < cur_idx:
              note_chords = note_chords + \
                        [NO_CHORD] * (cur_idx - len(note_chords))
          if len(note_chords) == cur_idx:
              note_chords.append(chord)
          elif len(note_chords) == cur_idx + 1 and note_chords[-1] == NO_CHORD:
               note_chords[-1] = chord

      for i in range(1, len(note_chords)):
          if note_chords[i] == NO_CHORD:
              note_chords[i] = note_chords[i-1]

      midi_obj = encoding_to_midi(e)
      midi_obj.tempo_changes[0].tempo = 80
      midi_obj.instruments[0].notes.sort(key=lambda x: (x.start, -x.end))

      ticks = midi_obj.ticks_per_beat
      midi_obj.instruments[0].name = 'melody'
      midi_obj.instruments.append(miditoolkit.Instrument(program=0, is_drum=False, name='chord'))  # piano 33
      midi_obj.instruments[0].program = 40#24  # violin(40)
      midi_obj.instruments[1].notes = []

      lyrics = []
      for word in tmp:
          if word not in [',', '.']:
              lyrics.append(word)
          else:
              lyrics[-1] += word

      note_idx = 0

      word_idx = 0
      for idx, word in enumerate(lyrics):
          if word not in [',', '.']:
            note = midi_obj.instruments[0].notes[align_idxs[word_idx]]
            midi_obj.lyrics.append(miditoolkit.Lyric(text=word, time=note.start))
            word_idx += 1
          else:
              midi_obj.lyrics[-1].text += word
      for idx, chord in enumerate(note_chords):
          if chord != NO_CHORD:
              root, type = chord.split(':')
              root = pitch_dict[root]
              midi_obj.instruments[1].notes.append(miditoolkit.Note(velocity=80, pitch=C2 + root, start=(idx * 2) * ticks, end=(idx * 2 + 2) * ticks))
              for shift in _CHORD_KIND_PITCHES[type]:
                  midi_obj.instruments[1].notes.append(miditoolkit.Note(velocity=80, pitch=C3 + (root + shift) % 12, start=(idx * 2) * ticks, end=(idx * 2 + 2) * ticks))

            #解析midi_obj里的notes，并处理成时长
      durations, pitchs = get_notes(midi_obj)
      print(durations)
      print(pitchs)
      
      return midi_obj, durations, pitchs
  
  


      
    
  