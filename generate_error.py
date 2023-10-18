import numpy as np
import os
import re
from unidecode import unidecode
from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    return args

def load_config(path):
    cfg = OmegaConf.load(path)
    return cfg

def teen_code(cfg, sentence, pivot):
    random = np.random.uniform(0,1,1)[0]
    new_sentence = str(sentence)
    if random > pivot:
        for word in cfg.noise.acronym.keys():
            if re.search(word, new_sentence):
                random2 = np.random.uniform(0,1,1)[0]
                if random2 < 0.5:
                    new_sentence = new_sentence.replace(word, cfg.noise.acronym[word])
        for word in cfg.noise.teencode.keys():
            if re.search(word, new_sentence):
                random3 = np.random.uniform(0,1,1)[0]
                if random3 < 0.05:
                    new_sentence = new_sentence.replace(word, cfg.noise.teencode[word])
        return new_sentence
    else:
        return sentence
    

def add_noise(cfg, sentence):
    pivot = cfg.pivot
    pivot1 = cfg.pivot1
    pivot2 = cfg.pivot2
    sentence = teen_code(cfg, sentence, pivot)
    noise_sentence =""
    i = 0
    while i < len(sentence):
        if sentence[i] not in cfg.noise.letters:
            noise_sentence +=sentence[i]
        else:
            random = np.random.uniform(0,1,1)[0]
            if random < pivot1:
                noise_sentence +=(sentence[i])
            elif random < pivot2:
                if sentence[i] in cfg.noise.typo.keys() and sentence[i] in cfg.noise.region_vowel.keys():
                    random2 = np.random.uniform(0,1,1)[0]
                    if random2 <= 0.4:
                        noise_sentence +=cfg.noise.typo[sentence[i]]
                    elif random2 < 0.8:
                        noise_sentence += cfg.noise.region_vowel[sentence[i]]
                    elif random < 0.95:
                        noise_sentence +=unidecode(sentence[i])
                    else:
                        noise_sentence += sentence[i]

            elif sentence[i] in cfg.noise.region_vowel.keys():
                random4 = np.random.uniform(0,1,1)[0]
                if random4<=0.6:
                    noise_sentence+=cfg.noise.region_vowel[sentence[i]]
                elif random4<0.85 :
                    noise_sentence+=unidecode(sentence[i])                        
                else:
                    noise_sentence+=sentence[i]
            elif i<len(sentence)-1 :
                if sentence[i] in cfg.noise.region_character.keys() and (i==0 or sentence[i-1] not in cfg.noise.letters) and sentence[i+1] in cfg.noise.vowel:
                    random5=np.random.uniform(0,1,1)[0]
                    if random5<=0.9:
                        noise_sentence+=cfg.noise.region_character[sentence[i]]
                    else:
                            noise_sentence+=sentence[i]
                else:
                        noise_sentence+=sentence[i]

            else:
                new_random = np.random.uniform(0,1,1)[0]
                if new_random <=0.33:
                    if i == (len(sentence) - 1):
                        continue
                    else:
                        noise_sentence+=(sentence[i+1])
                        noise_sentence+=(sentence[i])
                        i += 1
                elif new_random <= 0.66:
                    random_letter = np.random.choice(np.array(cfg.noise.letters_latin).flatten(), 1)[0]
                    noise_sentence+=random_letter
                else:
                    pass
        i += 1
    return noise_sentence



# if __name__ =="__main__":
# print("Nhập đoạn text cần tạo noise")
# cfg = load_config("/home/tienvh/hoang_uet/NLP/src/configs/defaults.yaml")
# text=input()
# print(add_noise(cfg, text))












