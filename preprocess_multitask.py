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

from turtle import pos
from sentence_transformers import SentenceTransformer, util

# 【note】modify transformers version:
# sentence_bert: transformers-2.1.1
# bob: transformers-4.21.2
 

model = SentenceTransformer('all-MiniLM-L6-v2')


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
                    per_group+=(line[16:]+'\t')
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


def best_persona(persona_list, response):

    sentences1 = response
    sentences2 = persona_list

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    cosine_scores = cosine_scores.cpu().tolist()[0]

    li = list(zip(sentences2, cosine_scores))
    order_list = sorted(li, key=lambda x:x[1])
    pos_ord = [sent for sent, _ in order_list]
   
    return pos_ord[-1].strip()


def preprocess(args):
    print(f"Reading {args.dataset_type} dataset...")
    test_persona, test_query, test_response = read_convai2_split(args.testset) if args.dataset_type=='convai2' else read_ecdt2019_split(args.testset, split_type='test')
    assert len(test_persona) == len(test_query) == len(test_response)
    print("testset loaded.")

    gold_persona = []
    for pers, resp in tqdm(list(zip(test_persona, test_response))):
        persona_list = pers.strip().split('\t')
        assert len(persona_list) == 5 or len(persona_list) == 4 or len(persona_list) == 3, (persona_list)
        gold_persona.append(best_persona(persona_list, resp))
    assert len(test_persona) == len(test_query) == len(test_response) == len(gold_persona)
      

    path = f'./data/ConvAI2/convai2_tokenized_task1/' 
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + f'{args.split}.txt','w') as w:
        for p, h, r in zip(test_persona, test_query, test_response):
            task = 'task is dialogue generation .'
            p_part = ''
            for i, cur in enumerate(p.split('\t')):
                p_part += f' persona {i + 1} is {cur.strip()}'

            context = ' dialogue is ' + h.strip() # single turn
            response = ' the agent should reply with ' + ' & ' + r.strip()

            line = task + p_part + context + response
            w.write(line.strip() + '\n')


    path = f'./data/ConvAI2/convai2_tokenized_task2/' 
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + f'{args.split}.txt','w') as w:
        for p, h, r in zip(test_persona, test_query, gold_persona):
            task = 'task is choosing the best persona .'
            p_part = ''
            for i, cur in enumerate(p.split('\t')):
                p_part += f' persona {i + 1} is {cur.strip()}'

            context = ' dialogue is ' + h.strip() # single turn
            response = ' the best persona is ' + ' & ' + r.strip()

            line = task + p_part + context + response
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

    parser.add_argument("--split", type=str, default='train')

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
