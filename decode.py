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


import torch
import random as rd
import json

from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader



CUDA_AVAILABLE = False
if torch.cuda.is_available():
    CUDA_AVAILABLE = True
    print("CUDA IS AVAILABLE")
else:
    print("CUDA NOT AVAILABLE")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_tokenier_and_model(tokenizer, model):
    ########## set special tokens
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 32
    model.config.min_length = 3
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 1.0
    model.config.num_beams = 1
    model.config.temperature = 0.95
    model.config.output_hidden_states = True
    return tokenizer, model


def prepare_data_batch(batch): # 这块如何准备。。。
    persona_input_ids = batch['persona']['input_ids']
    persona_attention_mask = batch['persona']['attention_mask']
    persona_type_ids = batch['persona']['token_type_ids'] * 0 + 1

    query_input_ids = batch['query']['input_ids']
    query_attention_mask = batch['query']['attention_mask']
    query_type_ids = batch['query']['token_type_ids'] * 0
    # for i in range(len(query_type_ids)):
    #     for j in range(len(query_type_ids[i])):
    #         query_type_ids[i][j] = 1 - query_type_ids[i][j]

    input_ids = torch.cat([persona_input_ids, query_input_ids], -1)
    attention_mask = torch.cat([persona_attention_mask, query_attention_mask], -1)
    type_ids = torch.cat([persona_type_ids, query_type_ids], -1)

    decoder_input_ids = batch['response']['input_ids']
    decoder_attention_mask = batch['response']['attention_mask']
    mask_flag = torch.Tensor.bool(1 - decoder_attention_mask)
    lables = decoder_input_ids.masked_fill(mask_flag, -100)

    return input_ids, attention_mask, type_ids, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids


def predict(args):
    print("Load tokenized data...\n")
    tokenizer = BertTokenizer.from_pretrained(args.encoder_model)

    # Dataset, DataLoader
    test_dataset = ConvAI2Dataset(test_persona_tokenized,
                                  test_query_tokenized,
                                  test_response_tokenized,
                                  device) if args.dataset_type == 'convai2' else ECDT2019Dataset(test_persona_tokenized,
                                                                                                 test_query_tokenized,
                                                                                                 test_response_tokenized,                                                                                             device)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Loading Model
    print("Loading Model from %s" % model_path)
    model_path = f"./checkpoints/ConvAI2_lex/bertoverbert_{args.eval_epoch}"
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    tokenizer, model = set_tokenier_and_model(tokenizer, model)

    print(f"Writing generated results to {args.save_result_path}...")
    with open(args.save_result_path, "w", encoding="utf-8") as outf:
        for test_batch in tqdm(test_loader):

            input_ids, attention_mask, type_ids, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_data_batch(
                test_batch)

            generated = model.generate(input_ids,
                                       token_type_ids=type_ids,
                                       attention_mask=attention_mask,
                                       num_beams=args.beam_size,
                                       length_penalty=args.length_penalty,
                                       min_length=args.min_length,
                                       no_repeat_ngram_size=args.no_repeat_ngram_size,
                                       per_input_ids=persona_input_ids)
            generated_2 = model.generate(input_ids,
                                         token_type_ids=type_ids,
                                         attention_mask=attention_mask,
                                         num_beams=args.beam_size,
                                         length_penalty=args.length_penalty,
                                         min_length=args.min_length,
                                         no_repeat_ngram_size=args.no_repeat_ngram_size,
                                         use_decoder2=True,
                                         per_input_ids=persona_input_ids)
            # print ('generated')
            # print (generated)
            # print ('generated_2')
            # print (generated_2)

            # for resp_ids in generated_2:
            #     print ('resp_ids = ', resp_ids)
            #     print ('decoded = ', tokenizer.decode(resp_ids))

            generated_token = tokenizer.batch_decode(
                generated, skip_special_tokens=True)

            generated_token_2 = tokenizer.batch_decode(
                generated_2, skip_special_tokens=True)

            query_token = tokenizer.batch_decode(
                query_input_ids, skip_special_tokens=True)

            gold_token = tokenizer.batch_decode(decoder_input_ids,
                                                skip_special_tokens=True)

            persona_token = tokenizer.batch_decode(
                persona_input_ids, skip_special_tokens=True)

            
            for p, q, g, r, r2 in zip(persona_token, query_token, gold_token, generated_token, generated_token_2):
                outf.write(f"persona:{p}\tquery:{q}\tgold:{g}\tresponse_from_d1:{r}\tresponse_from_d2:{r2}\n")


if __name__ == "__main__":
    parser = ArgumentParser("Transformers EncoderDecoderModel")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--word_stat", action="store_true")
    parser.add_argument("--train_valid_split", type=float, default=0.1)

    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/bertoverbert_epoch_5")

    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)

    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument("--eval_epoch", type=int, default=7)
    parser.add_argument("--print_frequency", type=int, default=-1)
    parser.add_argument("--warm_up_steps", type=int, default=6000)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--min_length", type=int, default=3)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warm_up_learning_rate", type=float, default=3e-5)

    parser.add_argument("--save_result_path",
                        type=str,
                        default="test_result.tsv")
    parser.add_argument("--dataset_type",
                        type=str,
                        default='convai2')  # convai2, ecdt2019
    parser.add_argument("--ppl_type",
                        type=str,
                        default='sents')  # sents, tokens

    parser.add_argument("--local_rank", type=int, default=0)
    
    '''
    dumped_token
        convai2:    ./data/ConvAI2/convai2_tokenized/
        ecdt2019:   ./data/ECDT2019/ecdt2019_tokenized/
    '''
    parser.add_argument("--dumped_token",
                        type=str,
                        default=None,
                        required=True)
    args = parser.parse_args()

    predict(args)


