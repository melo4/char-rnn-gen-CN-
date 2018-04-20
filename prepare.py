# -*- coding: utf-8 -*-
import unidecode
import string
import random
import re
import time
import torch
import math
from torch.autograd import Variable

# all_characters = string.printable  # 可打印字符
# n_characters = len(all_characters)
with open('./jay.txt',encoding='utf-8') as f:
    ldata = f.read()
all_characters = list(set(ldata))
all_characters.sort()
n_characters = len(all_characters)

def read_file(filename):
    file = (open(filename,encoding='utf-8').read())
    return file,len(file)


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

chunk_len =10
def random_chunk():
    start_index = random.randint(0,file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

# print(random_chunk())

def time_since(since):
    s = time.time() - since
    m = math.floor(s /60)
    s -= m*60
    return '%dm %ds' % (m, s)