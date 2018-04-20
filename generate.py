# -*- coding: utf-8 -*-
import torch
from prepare import *
from model import *

def generate(decoder,prime_str='你要',predict_len=500,temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    for p in range(len(prime_str)-1):
        _,hidden = decoder(prime_input[p], hidden)

    inp = prime_input[-1]

    for p in range(predict_len):
        output,hidden = decoder(inp,hidden)

        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist,1)[0]

        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char) #输出作为下一个输入

    return predicted

# main
if __name__ == '__main__':
    import argparse
    argparse = argparse.ArgumentParser()
    argparse.add_argument('filename', type=str)
    argparse.add_argument('-p', '--prime_str', type=str, default='A')
    argparse.add_argument('-l', '--predict_len', type=int, default=500)
    argparse.add_argument('-t', '--temperature', type=float, default=0.8)
    args = argparse.parse_args()

    decoder = torch.load(args.filename)
    del args.filename
    print(generate(decoder, **vars(args)))