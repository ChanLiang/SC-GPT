import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium')

checkpoint = torch.load('persona-output-dg-task-resaved/pytorch_model.bin')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
