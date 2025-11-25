import torch
from pytorch_pretrained_bert.modeling import (CONFIG_NAME, WEIGHTS_NAME, BertConfig, BertPreTrainedModel, BertModel)

checkpoint = '/workspace/tangxuemei/pre_trained_models/guwen_roberta/'
model= BertModel.from_pretrained(checkpoint)

model.load_state_dict(torch.load(checkpoint))
model.eval()
model_path = 'new_pytorch.bin'
torch.save(model.state_dict(), model_path, use_new_zipfile_serialization=False)
