from Melody.seq2seq.get_melody import LyricMelody

if __name__ == '__main__':
  
  model = LyricMelody()
  lyrics = input(">>")
  lyrics, notes, durations = model.get_melody(lyrics)
  print(lyrics)
  print(notes)
  print(durations)
  # en_vocab_file = os.path.join(root_path, 'Melody/checkpoints/seq2seq/lyric2note/best/en_vocab.pkl')
  # with open(en_vocab_file, 'rb') as f:
  #   data = pickle.load(f)
  #   print(data)