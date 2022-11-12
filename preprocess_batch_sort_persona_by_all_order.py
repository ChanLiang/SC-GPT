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


def sort_persona(persona_list, response):

    sentences1 = response
    sentences2 = persona_list

    #Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    # print (cosine_scores.cpu().tolist()[0]) # n x m 
    cosine_scores = cosine_scores.cpu().tolist()[0]

    li = list(zip(sentences2, cosine_scores))
    # order_list = sorted(li, key=lambda x:x[1], reverse=True)
    order_list = sorted(li, key=lambda x:x[1])
    # print ('=' * 60)
    # print (resp)
    # print (order_list)
    # print ()

    pos_ord = [sent for sent, _ in order_list]
    neg_ord = pos_ord[::-1]
    lex_pos_ord = sorted(pos_ord, key=str.lower)
    lex_neg_ord = lex_pos_ord[::-1]
    
    pos_maj3, pos_maj10, neg_maj3, neg_maj10 = [], [], [], []
    max_score, min_score = max(cosine_scores), min(cosine_scores)
    for pair in li:
        pos_maj3.append(pair[0])
        pos_maj10.append(pair[0])
        neg_maj3.append(pair[0])
        neg_maj10.append(pair[0])

        if max_score == pair[1]:
            pos_maj3 += [pair[0]] * 2
            pos_maj10 += [pair[0]] * 9

        if min_score == pair[1]:
            neg_maj3 += [pair[0]] * 2
            neg_maj10 += [pair[0]] * 9

    assert 6 <= len(pos_maj3) <= 7, len(pos_maj3)
    assert 6 <= len(neg_maj3) <= 7, len(neg_maj3)

    assert 13 <= len(pos_maj10) <= 14, len(pos_maj10)
    assert 13 <= len(neg_maj10) <= 14, len(neg_maj10)

    ret = [persona_list, pos_ord, neg_ord, lex_pos_ord, lex_neg_ord, pos_maj3, pos_maj10, neg_maj3, neg_maj10, [pos_ord[-1]], [pos_ord[-1]] * 5]
    # return [' '.join(item) for item in ret]
    return ret


def preprocess(args):
    print(f"Reading {args.dataset_type} dataset...")
    test_persona, test_query, test_response = read_convai2_split(args.testset) if args.dataset_type=='convai2' else read_ecdt2019_split(args.testset, split_type='test')
    assert len(test_persona) == len(test_query) == len(test_response)
    test_query_, test_response_ = test_query, test_response
    print("testset loaded.")

    print ('sorting persona...')
    res_list = [[] for _ in range(11)]
    for pers, resp in tqdm(list(zip(test_persona, test_response))):
        persona_list = pers.strip().split('\t')
        assert len(persona_list) == 5 or len(persona_list) == 4, (persona_list)
        ret = sort_persona(persona_list, resp)
        for i in range(11):
            res_list[i].append(ret[i])

    suffix_list = ['normal_ord', 'pos_ord', 'neg_ord', 'lex_pos_ord', 'lex_neg_ord', 'pos_maj3', 'pos_maj10', 'neg_maj3', 'neg_maj10', 'single_pos', 'multi_pos']
    for suffix, cur_persona in zip(suffix_list, res_list): # 11 types
        print (suffix)
        test_persona = cur_persona
        test_query, test_response = test_query_, test_response_

        assert len(test_persona) == len(test_query) == len(test_response), (len(test_persona), len(test_query), len(test_response))

        path = f'./data/ConvAI2/convai2_tokenized_{suffix}/' 
        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Saving tokenized dict at {path}")

        with open(path + f'test_{suffix}.txt','w') as w:
            for p, h, r in zip(test_persona, test_query, test_response):
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
