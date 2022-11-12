import torch
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('persona-output-dg-task/checkpoint-5000')

torch.save(
            {'model_state_dict': model.state_dict()}, 
            # 'persona-output-dg-task-resaved/pytorch_model.bin',
            'persona-output-dg-task/checkpoint-5000-resaved/pytorch_model_resaved.bin',
            _use_new_zipfile_serialization=False
        )
