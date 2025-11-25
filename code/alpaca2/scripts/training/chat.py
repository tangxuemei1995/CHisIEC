import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
# with open('/ceph/home/jun01/tangxuemei/Alpaca2/scripts/training/test_data/test.json',encoding='utf-8') as f:
#     line=f.readline() 
path = '/ceph/home/jun01/tangxuemei/Alpaca2/alpaca_chisre_re_epoch=30'
test_file = '/ceph/home/jun01/tangxuemei/Alpaca2/scripts/training/data/chisre/test.json'   
model = AutoModelForCausalLM.from_pretrained(path)
# model = AutoModelForCausalLM.from_pretrained('ziqingyang/chinese-alpaca-2-chat-7b',device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =model.cuda()
model =model.eval()

tokenizer = AutoTokenizer.from_pretrained(path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
w = open(path+'/test_result.txt','w',encoding='utf-8')
# DEFAULT_SYSTEM_PROMPT='你是一个语义关系分类的工具'
DEFAULT_SYSTEM_PROMPT = ''
TEMPLATE = (
    "[INST] <<SYS>>"
    "{system_prompt}"
    "<</SYS>>"
    "{instruction} [/INST]"
)

def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

for line in open(test_file,encoding='utf-8'):
    line=json.loads(line)
    input_text = generate_prompt(line['instruction'] + line['input'])
    input_ids = tokenizer(input_text, return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
    generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":1500,
    "do_sample":True,
    "top_k":40,
    "top_p":0.9,
    "temperature":0.5,
    "repetition_penalty":1.1,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
    }

    generate_ids  = model.generate(**generate_input)
    text = tokenizer.decode(generate_ids[0])   
    # with open('/ceph/home/jun01/tangxuemei/Alpaca2/scripts/alpaca_re-combined/test_result.txt','w',encoding='utf-8') as w:
    w.write('model output:\t' + text +'\t\t' + 'true label:\t' + line['output'] +'\n')
    # print(text)
    # exit()