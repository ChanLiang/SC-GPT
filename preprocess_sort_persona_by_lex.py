# Copyright 2021 Haoyu Song
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
import os
from argparse import ArgumentParser
# from dataloader import read_convai2_split
# from turtle import pos

def read_convai2_split(split_dir):
    persona = []
    query = []
    response = []
    try:
        with open(split_dir, "r", encoding="utf-8") as src:
            pre_st, st = 'dia', 'dia'
            for line in src:
                line = line.strip()
                if 'your persona:' in line:
                    pre_st = st
                    st = 'per'
                else:
                    pre_st = st
                    st = 'dia'

                if pre_st == 'dia' and st == 'per':
                    per_group = ''

                if st == 'per':
                    # per_group+=(line[16:]+' ')
                    per_group+=(line[16:]+'\t\t')
                elif st == 'dia':
                    persona.append(per_group)
                    line = line[line.find(' '):]
                    query.append(line.split('\t')[0])
                    response.append(line.split('\t')[1])
                else:
                    raise (ValueError)
    except FileNotFoundError:
        print(f"Sorry! The file {split_dir} can't be found.")
    return persona, query, response


def preprocess(args):
    print(f"Reading {args.dataset_type} dataset...")
    test_persona, test_query, test_response = read_convai2_split(args.testset) if args.dataset_type=='convai2' else read_ecdt2019_split(args.testset, split_type='test')
    assert len(test_persona) == len(test_query) == len(test_response)
    print (len(test_persona))

    n_persona = []
    for persona in test_persona:
        persona_list = persona.strip().split('\t\t')
        assert len(persona_list) == 4 or len(persona_list) == 5 or len(persona_list) == 3, len(persona_list)
        lex_pos_ord = sorted(persona_list, key=str.lower)
        n_persona.append(lex_pos_ord)
    
    assert len(n_persona) == len(test_query) == len(test_response)

    path = f'./data-bob/' 
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'train_lex_pos.txt','w') as w:
        for p, h, r in zip(n_persona, test_query, test_response):
            persona = ' persona: ' + ' persona: '.join(p)
            context, response = ' <s> ' + h.strip() + " <s> ", r.strip()
            line = persona.strip() + context + ' & ' + response.strip()
            w.write(line.strip() + '\n')


if __name__ == "__main__":
    parser = ArgumentParser("Transformers EncoderDecoderModel Preprocessing")
    parser.add_argument(
        "--trainset",
        type=str,
        default=
        "./data/ConvAI2/train_self_original_no_cands.txt")
    parser.add_argument(
        "--testset",
        type=str,
        default=
        "./data/ConvAI2/valid_self_original_no_cands.txt")
    parser.add_argument(
        "--nliset",
        type=str,
        default=
        "./data/ConvAI2/")
    
    parser.add_argument("--roberta", action="store_true")

    parser.add_argument("--train_valid_split", type=float, default=0.1)
    parser.add_argument("--max_source_length", type=int, default=32)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument(
        "--encoder_model_name_or_path",
        type=str,
        default="./pretrained_models/bert/bert-base-uncased")

    parser.add_argument("--dataset_type",
                        type=str,
                        default='convai2',
                        required=True)  # convai2, ecdt2019

    args = parser.parse_args()
    
    print ('start preprocess...')
    preprocess(args)
