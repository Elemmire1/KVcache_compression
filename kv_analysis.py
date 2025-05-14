import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import json

# 参数设置
LAYER_INTERVAL = 1
TOTAL_LAYERS = 32
LAYERS_TO_ANALYZE = list(range(0, TOTAL_LAYERS, LAYER_INTERVAL))
NUM_SAMPLES = 101
TOKENS_TO_COLLECT = 2048

# 加载模型和 tokenizer
model_path = "./qwen-7b"
tokenizer = AutoTokenizer.from_pretrained(
    model_path, pad_token='<|extra_0|>', eos_token='<|endoftext|>', padding_side='left', trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_path, pad_token_id=tokenizer.pad_token_id, trust_remote_code=True, device_map="auto"
).eval()

# Hook 函数用于捕获 KV
layer_kvs = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        #print(114514)
        if hasattr(module, 'attn'):
            hidden_states = input[0]  # [batch, seq_len, hidden_size]
            # 手动获取注意力参数
            q_proj = module.attn.c_attn.weight[:model.config.hidden_size, :]
            k_proj = module.attn.c_attn.weight[model.config.hidden_size:2*model.config.hidden_size, :]
            v_proj = module.attn.c_attn.weight[2*model.config.hidden_size:, :]
            bias = module.attn.c_attn.bias  # [3*hidden_size]
            k_bias = bias[model.config.hidden_size:2*model.config.hidden_size]
            v_bias = bias[2*model.config.hidden_size:]

            # Linear projection
            k = torch.matmul(hidden_states, k_proj.T) + k_bias
            v = torch.matmul(hidden_states, v_proj.T) + v_bias
            # shape: [batch, seq_len, hidden_size] -> 去掉 batch
            k = k[0].detach().cpu()
            v = v[0].detach().cpu()
            layer_kvs[layer_idx] = (k, v)
    return hook


# 注册 Hook
handles = []
for layer_idx in LAYERS_TO_ANALYZE:
    #block = model.transformer.layers[layer_idx]
    block = model.transformer.h[layer_idx]
    handles.append(block.register_forward_hook(make_hook(layer_idx)))

# 加载数据集并选取样本
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", cache_dir="./data")
samples = random.sample(list(dataset["validation"]), NUM_SAMPLES)

all_layer_avg_diffs = {}
eps = 1e-6

for sample in tqdm(samples, desc="Processing samples"):
    prompt = sample['article'][:3048] + "，是一段英文新闻 它的中文翻译是："
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=3048).to(model.device)

    # 清空旧 kv
    layer_kvs.clear()
    with torch.no_grad():
        model(**inputs)

    #print(LAYERS_TO_ANALYZE)
    #print(layer_kvs)
    for layer_idx in LAYERS_TO_ANALYZE:
        # if layer_idx not in layer_kvs:
        #     continue
        k, v = layer_kvs[layer_idx]  # [seq_len, hidden_size]
        # if k.shape[0] < TOKENS_TO_COLLECT:
        #     continue

        k = k[:TOKENS_TO_COLLECT]
        v = v[:TOKENS_TO_COLLECT]

        # 归一化差值计算
        # k_diff = (k[:-1] - k[1:]) / (0.5 * (torch.norm(k[:-1], dim=1) + torch.norm(k[1:], dim=1)).unsqueeze(1) + eps)
        # v_diff = (v[:-1] - v[1:]) / (0.5 * (torch.norm(v[:-1], dim=1) + torch.norm(v[1:], dim=1)).unsqueeze(1) + eps)
        k_diff = (k[:-1] - k[1:]) / (torch.norm(k,dim=1).mean()+ eps)
        v_diff = (v[:-1] - v[1:]) / (torch.norm(v,dim=1).mean()+ eps)

        #print(k_diff , "114\n")

        # 平均后累加
        k_mean = torch.abs(k_diff).mean(dim=0)
        v_mean = torch.abs(v_diff).mean(dim=0)

        # if layer_idx not in all_layer_avg_diffs:
        #     all_layer_avg_diffs[layer_idx] = {'k': [], 'v': []}

        if layer_idx not in all_layer_avg_diffs:
            all_layer_avg_diffs[layer_idx] = {'k': None, 'v': None, 'count': 0}

        all_layer_avg_diffs[layer_idx]['count'] += 1
        count = all_layer_avg_diffs[layer_idx]['count']
        k_mean = k_mean.to(torch.float32).numpy()
        v_mean = v_mean.to(torch.float32).numpy()
        # all_layer_avg_diffs[layer_idx]['k'].append(k_mean.numpy())
        # all_layer_avg_diffs[layer_idx]['v'].append(v_mean.numpy())

        if all_layer_avg_diffs[layer_idx]['k'] is None:
            all_layer_avg_diffs[layer_idx]['k'] = k_mean
            all_layer_avg_diffs[layer_idx]['v'] = v_mean
        else:
        # 增量更新平均值
            all_layer_avg_diffs[layer_idx]['k'] += (k_mean - all_layer_avg_diffs[layer_idx]['k']) / count
            all_layer_avg_diffs[layer_idx]['v'] += (v_mean - all_layer_avg_diffs[layer_idx]['v']) / count

        # all_layer_avg_diffs[layer_idx]['k']=all_layer_avg_diffs[layer_idx]['k'].mean(axis=0)
        # all_layer_avg_diffs[layer_idx]['v']=all_layer_avg_diffs[layer_idx]['v'].mean(axis=0)

# 注销 hook
for h in handles:
    h.remove()

sorted_k_indices = {}
sorted_v_indices = {}
# 聚合结果 + 画图
for layer_idx in LAYERS_TO_ANALYZE:
    # Key 处理
    # k_all = np.stack(all_layer_avg_diffs[layer_idx]['k'])  # [num_samples, dim]
    # k_avg = k_all.mean(axis=0)
    k_avg = all_layer_avg_diffs[layer_idx]['k']
    k_sorted = np.sort(k_avg)
    k_indices = np.argsort(k_avg)
    
    sorted_k_indices[layer_idx] = k_indices.tolist()
    # 写入文件
    with open("k_indices_sorted_by_layer.json", "w") as f:
        json.dump(sorted_k_indices, f, indent=2)

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(k_sorted)), k_sorted, color='skyblue')
    plt.title(f"Layer {layer_idx} - Sorted Key Dimension Diffs")
    plt.xlabel("Sorted Dimension Index")
    plt.ylabel("Avg Relative Diff")
   # plt.tight_layout()
    plt.savefig(f"figs1/layer_{layer_idx}_sorted_k_diff_bar.png")
    plt.close()

    # Value 处理
    # v_all = np.stack(all_layer_avg_diffs[layer_idx]['v'])  # [num_samples, dim]
    # v_avg = v_all.mean(axis=0)
    v_avg = all_layer_avg_diffs[layer_idx]['v']
    v_sorted = np.sort(v_avg)
    v_indices = np.argsort(v_avg)
    sorted_v_indices[layer_idx] = v_indices.tolist()
    # 写入文件
    with open("v_indices_sorted_by_layer.json", "w") as f:
        json.dump(sorted_v_indices, f, indent=2)
    
    
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(v_sorted)), v_sorted, color='skyblue')
    plt.title(f"Layer {layer_idx} - Sorted Value Dimension Diffs")
    plt.xlabel("Sorted Dimension Index")
    plt.ylabel("Avg Relative Diff")
   # plt.tight_layout()
    plt.savefig(f"figs1/layer_{layer_idx}_sorted_v_diff_bar.png")
    plt.close()

    print(f"Layer {layer_idx} top-5 dims: {np.argsort(k_avg)[:5]}")
