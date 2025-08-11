# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/8/12
# Description:


class PhonemeEncoder:
    def __init__(self, ):
        self.phoneme_list = [" ", "SHARP", "AP", "SP", "LP", "RP", "QS", "COLON", "u3", "ian3", "ao2", "uair4", "ou2", "anr4", "vr2", "ue2", "j", "uai2", "eng3", "ou1", "ang3", "uan3", "ue5", "inr4", "p", "iu4", "sh", "an1", "anr1", "ar4", "zh", "ing5", "t", "ai1", "uo1", "i5", "uan2", "g2", "o3", "e3", "ei2", "en2", "ui2", "a5", "l", "a1", "uai3", "i1", "w", "eng4", "ir4", "g5", "ei3", "er2", "uang3", "ang5", "n", "ai2", "iao2", "ar1", "i2", "ia2", "un4", "ei1", "ianr3", "iaor3", "u5", "ianr1", "in1", "ian4", "uai4", "iu2", "ao1", "engr4", "i,", "ir3", "uo3", "ue1", "ing4", "u2", "en5", "iur1", "iang4", "enr5", "aor3", "ar2", "ua3", "o1", "inr1", "ua4", "ir2", "uir3", "v4", "o5", "angr4", "eng2", "ie4", "d", "ur1", "uo2", "er3", "ing1", "ia4", "uang5", "ong4", "er1", "ui3", "an4", "o2", "e5", "s", "uanr4", "iao3", "er5", "uang2", "ia3", "an2", "uor1", "ing2", "ingr3", "c", "in4", "our5", "iao4", "enr2", "u4", "anr2", "uai1", "en3", "ong1", "ai5", "vr3", "v2", "ua1", "r", "en4", "enr3", "ou5", "ao5", "uor2", "ang1", "un3", "5", "ang4", "uan5", "uir4", "iang1", "ch", "un2", "ui1", "e,", "un1", "enr4", "ao3", "ou3", "e1", "air4", "ie2", "q", "engr1", "ianr2", "m", "ian2", "ue3", "z", "e4", "en1", "un5", "unr4", "e2", "uar1", "ang2", "uo5", "unr3", "er4", "ie1", "ar3", "k", "ei4", "v3", "ie3", "aor4", "ei5", "our2", "o4", "uo4", "ao4", "iu5", "an3", "in3", "eng5", "ong2", "uan4", "ian1", "ua5", "air2", "ai4", "iang5", "g4", "an5", "i3", "inr5", "uanr1", "or4", "iao1", "y", "ing3", "ve4", "uir5", "iang3", "eng1", "x", "a2", "ianr4", "uor3", "ong5", "g1", "ian5", "a4", "uan1", "ong3", "iong1", "ia5", "eir4", "ui5", "ui4", "iu3", "g3", "ai3", "iao5", "ie5", "a3", "in2", "ou4", "anr3", "b", "u1", "iong2", "ingr2", "iu1", "ue4", "in5", "iang2", "uang1", "ia1", "uang4", "i4", "ua2", "h", "f", "g", "inan2"]
        self.shengmu_set = set(['b','p','m','f','d','t','n','l','g','k','h','j','q','x','z','c','s','y','w','zh','ch','sh','r'])
        self.phoneme_dict = {s: i for i, s in enumerate(self.phoneme_list)}
        self.replace_dict = {
            ",": "SP",
            ".": "AP",
            "《": "LP",
            "》": "RP",
            "#": "SHARP",
            "？": "QS",
            "！": "COLON",
        }
        self.pause_set = set([p for p in self.replace_dict.values()])
        self.pause_set.add(" ")

    def encode(self, text):
        text = text.replace("ve", "ue")
        ids_seq = []
        len_text = len(text)
        for k in self.replace_dict:
            text = text.replace(k, self.replace_dict[k])

        for note in text.split(" "):
            if note in self.replace_dict.values():
                ids_seq += [self.phoneme_dict[note]]
            else:
                if note in self.phoneme_dict:
                    ids_seq += [self.phoneme_dict[note]] * len(note)
                else:
                    if note[:2] in self.shengmu_set:
                        shengmu = note[:2]
                        yunmu = note[2:]
                    else:
                        shengmu = note[:1]
                        yunmu = note[1:]
                    ids_seq += [self.phoneme_dict[shengmu]] * len(shengmu)
                    ids_seq += [self.phoneme_dict[yunmu]] * len(yunmu)
            ids_seq += [0]
        ids_seq = ids_seq[:-1]
        assert len_text == len(ids_seq), f"len(text)={len_text}, len(ids_seq)={len(ids_seq)}"
        return ids_seq


# text = "yao4 bu4 # , # ni3 men5 # liang3 ge4 # zai4 # hao3 hao5 # liao2 yi4 liao2 # ba5 # ."
# tool = PhonemeEncoder()
# print(tool.format_text(text))