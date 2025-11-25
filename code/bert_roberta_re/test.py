# 这个python的代码
import torch
torch.cuda.empty_cache()
# 计算一下总内存有多少。
total_memory = torch.cuda.get_device_properties(0).total_memory
print(total_memory)
# 占用全部显存:
tmp_tensor = torch.empty(int(total_memory), dtype=torch.int8, device='cuda')