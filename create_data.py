import os
import re
import random
import spacy
# Tạo một đối tượng nlp từ spaCy
nlp = spacy.load("en_core_web_sm")
from unidecode import unidecode
from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser
from string import ascii_letters, punctuation, digits
import numpy as np
import pandas as pd
import random 
from string import ascii_letters, punctuation, digits
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
from NLP.src.generate_spelling_Vietnamesetext.generate_error import add_noise
import warnings


"""Preprocsee input file"""
clean_chars = re.compile(r'[,!?’\'$%€\(\)\- ]', re.MULTILINE)
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False) # for vi


original_text = "This is an example text... with ellipses... in it."

def clean_dot(text):
    text_without_ellipses = re.sub(r'\.\.\.', '', text)
    return text_without_ellipses
    
def clean_text(text):
    # Define a regular expression pattern to remove unwanted characters
    cleaned_text = clean_chars.sub(' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text

def split_text_into_sentences(text):
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.(?![.]{3})|\?)\s'
    # bug: split sentence in case "..."
    sentences = re.split(pattern, text)
    return [sent.strip() for sent in sentences if sent.strip()]

    # take a very longer time :<
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp(text)
    # return [sent.text for sent in doc.sents]



def process_text_file(file_path):
    with open(file_path, encoding="utf16", errors='ignore') as txt_file:
        text = txt_file.readline()
        text = clean_text(text)
        text = clean_dot(text)
    return split_text_into_sentences(text)

# ========================================================================================================

"""
create dataset with spelling errors

"""
def load_config(path):
    cfg = OmegaConf.load(path)
    return cfg    

cfg = load_config("/home/tienvh/hoang_uet/NLP/src/configs/defaults.yaml")

def tokenizer_check_if_text_too_long(text, tokenizer, max_length):
    data = tokenizer.batch_encode_plus([text],max_length=max_length,truncation=True,return_overflowing_tokens=True )    
    if len(data["input_ids"]) > 1:
        return True
    else:
        return False


def delete_characters(text, char_delete_percentage=0.005):
    modifyed_line = []       
    for char in text:
        if random.random() > char_delete_percentage or char in digits:
            modifyed_line.append(char)
    return "".join(modifyed_line)


def insert_characters(text, augmentation_probability=0.01):
    modifyed_line = []   
    for char in text:
        if random.random() <= augmentation_probability and char not in digits:            
            modifyed_line.append(random.choice(ascii_letters))
        modifyed_line.append(char)
    return "".join(modifyed_line)


def replace_characters(text, augmentation_probability=0.01):
    modifyed_line = []   
    for char in text:
        if random.random() <= augmentation_probability and char not in digits:            
            modifyed_line.append(random.choice(ascii_letters))
        else:
            modifyed_line.append(char)
    return "".join(modifyed_line)


def swap_characters_case(text, augmentation_probability=0.01):
    modifyed_line = []   
    for char in text:
        if random.random() <= augmentation_probability:            
            char = char.swapcase()
        modifyed_line.append(char)
    return "".join(modifyed_line)


def lower_case_words(text, augmentation_probability=0.05):
    modifyed_line = []   
    for word in text.split():
        if word[0].islower() == False and random.random() <= augmentation_probability:            
            word = word.lower()
        modifyed_line.append(word)
    return " ".join(modifyed_line)

def delete_word(text, augmentation_probability = 0.001):        
    if random.random() < augmentation_probability:
        words = text.split()
        if len(words) < 3:
            # do not delete word in short text, as there will be no context to guess the word
            return text
        word_to_remove = random.randint(0,len(words)-1)
        words.pop(word_to_remove)
        return " ".join(words)
    else:
        return text
    
chars_regrex = cfg.noise.char_regrex

def _char_regrex(text):
    match_chars = re.findall(chars_regrex, text)
    return match_chars

def _random_replace(text, match_chars):
    replace_char = match_chars[np.random.randint(low=0, high=len(match_chars), size=1)[0]]
    insert_chars = cfg.noise.same_chars[unidecode.unidecode(replace_char)]
    insert_char = insert_chars[np.random.randint(low=0, high=len(insert_chars), size=1)[0]]
    text = text.replace(replace_char, insert_char, 1)
    return text

def change(text):
    match_chars = _char_regrex(text)
    if len(match_chars) == 0:
        return text
    text = _random_replace(text, match_chars)
    return text

def remove_empty(sentences):
    while("" in sentences):
        sentences.remove("")
        return sentences

clean_punctuation = re.compile(r"(?<!\d)[.,;:'?!](?!\d)")
def remove_punctuation(text):
    """Remove all punctuation from string, except if it's between digits"""
    return clean_punctuation.sub("", text)


def generate(text_line):
    noise_text = []
    for line in text_line:
        print(line)
        if tokenizer_check_if_text_too_long(line,tokenizer,max_length=1024):
            continue
        if random.random() > 0.02:
            new_line = swap_characters_case(line)  
            new_line = delete_word(new_line)                    
            new_line = delete_characters(new_line)
            new_line = insert_characters(new_line)
            new_line = replace_characters(new_line)
            new_line = add_noise(cfg, str(new_line))
            new_line = lower_case_words(new_line)                                           
            new_line = remove_punctuation(new_line)
        else:
            new_line = line
        print(new_line)
        noise_text.append(new_line)  
    with open('noise_text.txt', 'w', encoding='utf-8') as file:
        file.write('\n'.join(noise_text))
    return noise_text  


def count_newline_types(generated_lines, original_lines):
    newline_types_count = {}

    for generated_line, original_line in zip(generated_lines, original_lines):
        if generated_line == original_line:
            newline_type = "Original"
        else:
            newline_type = "Modified"

        if newline_type in newline_types_count:
            newline_types_count[newline_type] += 1
        else:
            newline_types_count[newline_type] = 1

    return newline_types_count


#=======================================================================================================
def main():

    folder_path = "/home/tienvh/hoang_uet/NLP/data/VNTC/Data/10Topics/Ver1.1/Train_Full/Train_Full"
    sentences = []
    # save text into singe sentence 
    # for root, _, files in os.walk(folder_path):
    #     for file in files:
    #         if file.endswith('.txt'):
    #             file_path = os.path.join(root, file)
    #             sentences.extend(process_text_file(file_path))
    #             # print(sentences)

    # with open("sentences.txt", "w") as file:
    #     for sent in sentences:

    #         file.write(sent + "\n")
    #######################################################
    
    # Generate error
    file = open('sentences.txt', 'r')
    Lines = file.readlines()
    incorrect_sentences = generate(Lines)
    return incorrect_sentences
 

if __name__ == "__main__":
    main()
