from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

# sc-gpt: transformers==2.1.1
# LSP: transformers==4.18.0

class TestTextSeqDataset(Dataset): # 当做纯文本处理了，没有用到 &
    def __init__(self, tokenizer, use_tokenize=True, file_path='', seperator=' & '):
        self.examples = []
        self.labels = []
        for line in open(file_path, 'r', encoding='utf-8'):
            raw_str = line.strip().lower()
            l, r = raw_str.split(seperator)

            if use_tokenize:
                tokenized_query = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(l))
                tokenized_response = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(r))
            else:
                tokenized_query = tokenizer.convert_tokens_to_ids(l.split())
                tokenized_response = tokenizer.convert_tokens_to_ids(r.split())
            
            self.examples.append(tokenized_query)
            self.labels.append(tokenized_response) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item]), torch.tensor(self.labels[item])


print ("load tokenizer and model...")
# tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
# model = AutoModelWithLMHead.from_pretrained('microsoft/DialoGPT-medium')
tokenizer = GPT2Tokenizer.from_pretrained('../dialogpt-models/')
model = GPT2LMHeadModel.from_pretrained('../dialogpt-models/')

# tokenizer = GPT2Tokenizer.from_pretrained('../SC-GPT/persona-output-dg-task/')
# model = GPT2LMHeadModel.from_pretrained('../SC-GPT/persona-output-dg-task/')

checkpoint = torch.load('persona-output-dg-task/checkpoint-5000-resaved/pytorch_model_resaved.bin')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# 1. single batch
# # line = "How about Morty?Is he still a single father?" + tokenizer.eos_token
# line = "task is dialogue generation . persona 1 is i like to remodel homes. persona 2 is i like to go hunting. persona 3 is i like to shoot a bow. persona 4 is my favorite holiday is halloween. persona 5 is  dialogue is hi , how are you doing ? i am getting ready to do some cheetah chasing to stay in shape . the agent should reply with "
# new_user_input_ids = tokenizer.encode(line, return_tensors='pt')
# # print(new_user_input_ids.shape) # [1, 6]
# bot_input_ids = new_user_input_ids.repeat((4, 1))
# # print (bot_input_ids.shape) # [8, 6]


# 2. dataloader
test_dataset = TestTextSeqDataset(tokenizer, file_path='data/ConvAI2/convai2_tokenized_task1/test.txt')
test_sampler = RandomSampler(test_dataset) 
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=8)

for step, batch in enumerate(test_dataloader):
    inputs, labels = batch
    bot_input_ids = inputs
    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=150,
        pad_token_id=tokenizer.eos_token_id,  
        no_repeat_ngram_size=3,  
        # num_beams=10,
        do_sample=True, 
        top_k=5, 
        top_p=0.9,
        temperature = 1.0
    )
    # print (chat_history_ids.shape) # torch.Size([8, 14])
    # pretty print last ouput tokens from bot
    # print("RickBot: {}".format(tokenizer.batch_decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    # print("RickBot: {}".format())
    resps = tokenizer.batch_decode(chat_history_ids[:, bot_input_ids.shape[-1] + 1:], skip_special_tokens=True)
    for e in resps:
        print (e.strip())
