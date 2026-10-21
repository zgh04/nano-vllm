import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# safe tensors 分片权重加载器
# 背景：HuggingFace 保存的权重文件中，每个矩阵是独立存储的：
#   q_proj.weight,k_proj.weight,v_proj.weight
#   gate_proj.weight,up_proj.weight
# 但 nano-vLLM 中为了性能，这些矩阵被融合成一个 Parameter：
#   self.qkv_proj.weight
#   self.gate_up_proj.weight

# qwen3.py, Qwen3ForCausalLM:
#packed_modules_mapping = {
#   "q_proj":      ("qkv_proj",     "q"),      # q_proj 的权重 → qkv_proj 的 Q 区域
#   "k_proj":      ("qkv_proj",     "k"),       # k_proj 的权重 → qkv_proj 的 K 区域
#   "v_proj":      ("qkv_proj",     "v"),       # v_proj 的权重 → qkv_proj 的 V 区域
#   "gate_proj":   ("gate_up_proj",  0),         # gate_proj 的权重 → gate_up_proj 的第 0 块
#   "up_proj":     ("gate_up_proj",  1),         # up_proj 的权重 → gate_up_proj 的第 1 块
# }
#每一条映射的含义：HuggingFace 中的 key → nano-vLLM 中的 param_name，以及它在融合矩阵中的位置（shard_id）。


# 默认权重加载器
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# model：示例化完成的空模型 权重随机初始化，等待加载
# path 存放一堆分片.safetensors 的文件夹路径
# 
#- load_model 解决"名称映射"问题
#- packed_modules_mapping 解决"融合矩阵的区域"问题
#- weight_loader 解决"TP 分片"问题
#- param 本身解决"按什么精度存在哪"问题
def load_model(model: nn.Module, path: str):
    # 合并层映射：大模型为了加速计算，会把多个独立权重打包合并存储
    # 如，注意力 QKV 原本是三个独立线性层，权重文件里会合并存为 qkv_proj
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})   # 定义在 model 里
    # 遍历所有 safetensors 文件
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            # 遍历当前分片内所有权重名称字符串，如 layers.0.qkv_proj.weight
            for weight_name in f.keys():
                # 当前权重属于 '合并打包层'
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        # 将文件里的打包名称替换成模型内部真实参数名，layers.0.qkv_proj.weight -> layers.0.q_proj.weight
                        param_name = weight_name.replace(k, v)
                        # 根据真实参数名从模型中取出对应的 nn.Parameter
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                # 普通权重，未触发 break 时进入 else
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
