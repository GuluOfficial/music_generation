import random
import miditoolkit
import math
import librosa


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
sample_len_max = 200  # window length max
sample_overlap_rate = 4
ts_filter = False
min_pitch = 48
max_pitch = 72
min_oct = 5
max_oct = 6

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
ts_dict = dict()
ts_list = list()
for i in range(0, max_ts_denominator + 1):  # 1 ~ 64
    for j in range(1, ((2 ** i) * max_notes_per_bar) + 1):
        ts_dict[(j, 2 ** i)] = len(ts_dict)
        ts_list.append((j, 2 ** i))


def adapt_e(e, align_idxs):
    tmp = [list(i) for i in e]
    last_pos = 0
    for i in range(len(tmp)):
        note = tmp[i]
        if note[3] <= min_oct * 12:
            note[3] = min_oct * 12 + note[3] % 12
        elif note[3] >= max_oct * 12 + 12:
            note[3] = max_oct * 12 + note[3] % 12

        # 16th note
        if note[1] % 2 == 1 and last_pos <= (16 * note[0] + note[1] - 1):
            note[1] -= 1
        if note[4] != 1 and (note[1] + note[4]) % 2 == 1:
            note[4] -= 1
        if last_pos >= 16 * note[0] + note[1]:
            tmp[i - 1][4] -= last_pos - (16 * note[0] + note[1])
        last_pos = 16 * note[0] + note[1] + note[4]
        tmp[i] = note
    # ensure no rest in a word:
    words = []
    cur_word = []
    for idx, note in enumerate(tmp):
        if idx != 0 and idx in align_idxs:
            assert len(cur_word)
            if len(cur_word):
                words.append(cur_word)
                cur_word = []
        cur_word.append(note)
    if len(cur_word):
        words.append(cur_word)
    tmp = []
    for notes in words:
        first_note = notes[0]
        last_pos = 16 * first_note[0] + first_note[1] + first_note[4]
        tmp.append(first_note)
        for note in notes[1:]:
            note[0] = last_pos // 16
            note[1] = last_pos % 16
            tmp.append(note)
            last_pos += note[4]

    # remove empty bar:
    last_pos = 0
    offset = 0
    for note in tmp:
        cur_pos = 16 * (note[0] + offset) + note[1]
        while cur_pos - last_pos >= 16:
            offset -= 1
            cur_pos -= 16
        note[0] += offset
        last_pos = cur_pos + note[4]

    tmp = [tuple(i) for i in tmp]
    return tmp


def adapt(pattern):
    num_dict = dict()
    prev = []
    for sent_idx, sent in enumerate(pattern):
        for sep_idx, sep in enumerate(sent):
            cur_len = len(sep)
            cur_starts = []
            if cur_len in num_dict and random.random() < 1.0:
                cur_starts = num_dict[cur_len]
                prev.extend(cur_starts)
                print('reuse rhythm:', cur_starts)
            else:
                if len(cur_starts) == 0:
                    offset = 0
                    if len(prev) and (sep[0] - prev[-1]) % 4 <= 1:
                        offset = sep[0] - prev[-1] + 2

                    cur_beats = (sep[0] - offset) % 4
                    new_sent = [cur_beats]
                    for item in sep[1:]:
                        if (item - offset - cur_beats) % 4 >= 2:
                            offset += (item - offset - cur_beats) % 4 - 1
                        if len(prev) >= 4 and len(set(prev[-4:])) == 1 and prev[-1] == (item - offset) % 4:
                            offset -= 1
                        new_sent.append((item - offset) % 4)
                        prev.append((item - offset) % 4)
                        cur_beats = new_sent[-1]

                    cur_starts = new_sent
                    num_dict[cur_len] = cur_starts
            pattern[sent_idx][sep_idx] = cur_starts
    return pattern


def enc_ts(x):
    assert x in ts_dict, 'unsupported time signature: ' + str(x)
    return ts_dict[x]


def dec_ts(x):
    return ts_list[x]


def enc_dur(x):
    return min(x, duration_max * pos_resolution)


def dec_dur(x):
    return x


def enc_vel(x):
    return x // velocity_quant


def dec_vel(x):
    return (x * velocity_quant) + (velocity_quant // 2)


def enc_tpo(x):
    x = max(x, min_tempo)
    x = min(x, max_tempo)
    x = x / min_tempo
    e = round(math.log2(x) * tempo_quant)
    return e


def dec_tpo(x):
    return 2 ** (x / tempo_quant) * min_tempo


def encoding_to_midi(encoding):
    bar_to_timesig = [list() for _ in range(max(map(lambda x: x[0], encoding)) + 1)]
    for i in encoding:
        bar_to_timesig[i[0]].append(i[6])
    bar_to_timesig = [max(set(i), key=i.count) if len(i) > 0 else None for i in bar_to_timesig]
    for i in range(len(bar_to_timesig)):
        if bar_to_timesig[i] is None:
            bar_to_timesig[i] = enc_ts(4, 4) if i == 0 else bar_to_timesig[i - 1]
    bar_to_pos = [None] * len(bar_to_timesig)
    cur_pos = 0
    for i in range(len(bar_to_pos)):
        bar_to_pos[i] = cur_pos
        ts = dec_ts(bar_to_timesig[i])
        measure_length = ts[0] * beat_note_factor * pos_resolution // ts[1]
        cur_pos += measure_length
    pos_to_tempo = [list() for _ in range(cur_pos + max(map(lambda x: x[1], encoding)))]
    for i in encoding:
        pos_to_tempo[bar_to_pos[i[0]] + i[1]].append(i[7])
    pos_to_tempo = [round(sum(i) / len(i)) if len(i) > 0 else None for i in pos_to_tempo]
    for i in range(len(pos_to_tempo)):
        if pos_to_tempo[i] is None:
            pos_to_tempo[i] = enc_tpo(80.0) if i == 0 else pos_to_tempo[i - 1]
    midi_obj = miditoolkit.midi.parser.MidiFile()

    def get_tick(bar, pos):
        return (bar_to_pos[bar] + pos) * midi_obj.ticks_per_beat // pos_resolution

    midi_obj.instruments = [
        miditoolkit.containers.Instrument(program=(0 if i == 128 else i), is_drum=(i == 128), name=str(i)) for i in
        range(128 + 1)]
    for i in encoding:
        start = get_tick(i[0], i[1])
        program = i[2]
        pitch = (i[3] - 128 if program == 128 else i[3])
        duration = get_tick(0, dec_dur(i[4]))
        end = start + duration
        velocity = dec_vel(i[5])
        midi_obj.instruments[program].notes.append(
            miditoolkit.containers.Note(start=start, end=end, pitch=pitch, velocity=velocity))
    midi_obj.instruments = [i for i in midi_obj.instruments if len(i.notes) > 0]
    cur_ts = None
    for i in range(len(bar_to_timesig)):
        new_ts = bar_to_timesig[i]
        if new_ts != cur_ts:
            numerator, denominator = dec_ts(new_ts)
            midi_obj.time_signature_changes.append(
                miditoolkit.containers.TimeSignature(numerator=numerator, denominator=denominator, time=get_tick(i, 0)))
            cur_ts = new_ts
    cur_tp = None
    for i in range(len(pos_to_tempo)):
        new_tp = pos_to_tempo[i]
        if new_tp != cur_tp:
            tempo = dec_tpo(new_tp)
            midi_obj.tempo_changes.append(miditoolkit.containers.TempoChange(tempo=tempo, time=get_tick(0, i)))
            cur_tp = new_tp
    return midi_obj


def process(lyrics):
    '''
    将输入的歌词处理成以空格为间隔，sep为分割的形式
    模型只支持这样对齐的输入
    '''
    content = ''
    for w in lyrics:
        if w in set(list(",.!，。！？?；;、")):
            content = content + '[sep] '
        else:
            content = content + w + ' '
    content = content + '[sep]'
    return content




def get_notes(midi_obj):
    '''解析midi_obbj对象获取音符'''
    notes = midi_obj.instruments[0].notes
    durations = []
    pitchs = []
    print(notes)

    for n in notes:
        pit = n.pitch
        note = librosa.midi_to_note(pit)
        pitchs.append(note)
        offset = (n.end - n.start) / (1.0 * 1000)
        durations.append(offset)

    return durations, pitchs



