from io import open
import re
import unicodedata
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.utils import shuffle

class EnglishSpanishDataset(Dataset):
    def __init__(self, words_df: pd.DataFrame, sentences_df: pd.DataFrame, transform=None, target_transform=None, max_sentence_length=30):
        self.words_df = words_df
        self.vocab_size = len(words_df)
        self.sentences_df = sentences_df
        self.transform = transform
        self.target_transform = target_transform
        self.max_sentence_length = max_sentence_length

    def __len__(self):
        return len(self.sentences_df)

    def get_word_index(self, word: str):
        return self.words_df[self.words_df[0]==word].index[0]

    def create_input_and_target_tensor(self, sentence: str):
        words = sentence.split(" ")
        input_tensor = torch.zeros(len(words))
        target_tensor = torch.zeros(len(words))
        for idx in range(len(words)):
            input_tensor[idx] = self.get_word_index(words[idx])
        target_tensor[:len(words)-1] = torch.clone(input_tensor[1:])
        target_tensor[-1] = self.get_word_index("<EOS>")
        return input_tensor.to(torch.int64), target_tensor.to(torch.int64)

    def create_target_tensor(self, input_tensor):
        # tensor = torch.zeros(len(input_tensor), 1, self.vocab_size)
        # for idx in range(1, len(input_tensor)):
        #     word_ref = input_tensor[idx]
        #     tensor[idx-1][0][word_ref] = 1
        # tensor[len(input_tensor)-1][0][self.get_word_index("<EOS>")] = 1 # EOS
        target_tensor = torch.zeros(len(input_tensor))
        target_tensor[:len(input_tensor)-1] = torch.clone(input_tensor[1:])
        target_tensor[-1] = self.get_word_index("<EOS>")
        return target_tensor
    
    def tensor_to_sentence(self, tensor: torch.Tensor) -> str:
        sentence = ""
        for word_index in tensor:
            word = self.words_df.iloc[int(word_index)].at[0]
            sentence += word
            sentence += " "
        return sentence.strip()
    
    def create_batch(self, sentence):
        words = sentence.split(" ")
        input_vector = torch.zeros(self.max_sentence_length)
        input_vector = input_vector.new_full(input_vector.size(), self.get_word_index("<EOS>"))
        target_vector = torch.zeros(self.max_sentence_length)
        target_vector = target_vector.new_full(target_vector.size(), self.get_word_index("<EOS>"))

        for idx in range(len(words)):
            input_vector[idx] = self.get_word_index(words[idx])

        target_vector[:len(words)-1] = torch.clone(input_vector[1:len(words)])

        return input_vector.to(torch.int64), target_vector.to(torch.int64)


    def __getitem__(self, idx: int):
        sentence = self.sentences_df.iloc[idx, 0]
        input_tensor, target_tensor = self.create_batch(sentence)
        # return input_tensor, target_tensor
        return input_tensor, target_tensor


# Functions for data cleaning
def train_test_val_split(df: pd.DataFrame, train:float, val:float, test:float, random_state=42):
    assert train + val + test == 1
    assert isinstance(df, pd.DataFrame)

    df = shuffle(df, random_state=random_state)
    train_set = df.iloc[:int(len(df)*train),:]
    val_set = df.iloc[int(len(df)*train):int(len(df)*(train+val)),:]
    test_set = df.iloc[int(len(df)*(train+val)):,:]
    
    return train_set, val_set, test_set


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z.!'?]+", r" ", s)
    return s.strip()


def parse_data(filename):
    file = open(filename, 'r', encoding='utf-8')
    word_dictionary = {}
    word_list = []
    count = 0

    while True:
        count += 1
        line = file.readline()
        pair = [normalizeString(s) for s in line.split('\t')[:2]]

        with open('pairs.csv', mode='a', encoding='utf-8') as writer:
            writer.write(','.join(pair)+'\n')

        for sentence in pair:
            for word in sentence.split(" "):
                if word in word_dictionary:
                    continue
                else:
                    word_list.append(word)
                    word_dictionary[word] = len(word_list)-1
                    with open('annotations_file.csv', mode='a', encoding='utf-8') as writer:
                        writer.write(','.join([word, str(word_dictionary[word])]) + '\n')

        if count % 20000 == 0: print(f'Processing line {count}')
        if not line: print("Done!"); break

    # Adds <EOS> word
    word_list.append("<EOS>")
    word_dictionary["<EOS>"] = len(word_list)-1
    with open('annotations_file.csv', mode='a', encoding='utf-8') as writer:
        writer.write(','.join(["<EOS>", str(word_dictionary["<EOS>"])]) + '\n')
    
    return word_dictionary, word_list


if __name__=='__main__':
    word_dictionary, word_list = parse_data("spa.txt")