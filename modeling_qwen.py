# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import pandas as pd
import json
import copy
import importlib
import math
import pathlib
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import warnings

from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from einops import rearrange
except ImportError:
    rearrange = None
from torch import nn

SUPPORT_CUDA = torch.cuda.is_available()
SUPPORT_BF16 = SUPPORT_CUDA and torch.cuda.is_bf16_supported()
SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
SUPPORT_TORCH2 = hasattr(torch, '__version__') and int(torch.__version__.split(".")[0]) >= 2


from .configuration_qwen import QWenConfig
from .qwen_generation_utils import (
    HistoryType,
    make_context,
    decode_tokens,
    get_stop_words_ids,
    StopWordsLogitsProcessor,
)


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "qwen"
_CONFIG_FOR_DOC = "QWenConfig"

QWen_PRETRAINED_MODEL_ARCHIVE_LIST = ["qwen-7b"]

_ERROR_BAD_CHAT_FORMAT = """\
We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
我们检测到您可能在使预训练模型（而非chat模型）进行多轮chat，因为您当前在generation_config指定的chat_format，并未设置为我们在对话中所支持的"chatml"格式。
如果您在直接使用我们从Huggingface提供的模型，请确保您在调用model.chat()时，使用的是"Qwen/Qwen-7B-Chat"模型（而非"Qwen/Qwen-7B"预训练模型）。
"""

_SENTINEL = object()
_ERROR_STREAM_IN_CHAT = """\
Pass argument `stream` to model.chat() is buggy, deprecated, and marked for removal. Please use model.chat_stream(...) instead of model.chat(..., stream=True).
向model.chat()传入参数stream的用法可能存在Bug，该用法已被废弃，将在未来被移除。请使用model.chat_stream(...)代替model.chat(..., stream=True)。
"""

_ERROR_INPUT_CPU_QUERY_WITH_FLASH_ATTN_ACTIVATED = """\
We detect you have activated flash attention support, but running model computation on CPU. Please make sure that your input data has been placed on GPU. If you actually want to run CPU computation, please following the readme and set device_map="cpu" to disable flash attention when loading the model (calling AutoModelForCausalLM.from_pretrained).
检测到您的模型已激活了flash attention支持，但正在执行CPU运算任务。如使用flash attention，请您确认模型输入已经传到GPU上。如果您确认要执行CPU运算，请您在载入模型（调用AutoModelForCausalLM.from_pretrained）时，按照readme说法，指定device_map="cpu"以禁用flash attention。
"""

apply_rotary_emb_func = None
rms_norm = None
flash_attn_unpadded_func = None
flash_attn_func = None

def _import_flash_attn():
    global apply_rotary_emb_func, rms_norm, flash_attn_unpadded_func, flash_attn_func
    try:
        from flash_attn.layers.rotary import apply_rotary_emb_func as __apply_rotary_emb_func
        apply_rotary_emb_func = __apply_rotary_emb_func
    except ImportError:
        logger.warn(
            "Warning: import flash_attn rotary fail, please install FlashAttention rotary to get higher efficiency "
            "https://github.com/Dao-AILab/flash-attention/tree/main/csrc/rotary"
        )

    try:
        from flash_attn.ops.rms_norm import rms_norm as __rms_norm
        rms_norm = __rms_norm
    except ImportError:
        logger.warn(
            "Warning: import flash_attn rms_norm fail, please install FlashAttention layer_norm to get higher efficiency "
            "https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm"
        )

    try:
        import flash_attn
        _flash_attn_func = None
        if not hasattr(flash_attn, '__version__'):
            from flash_attn.flash_attn_interface import flash_attn_unpadded_func as __flash_attn_unpadded_func
        else:
            if int(flash_attn.__version__.split(".")[0]) >= 2:
                if int(flash_attn.__version__.split(".")[1]) >= 1:
                    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
                from flash_attn.flash_attn_interface import flash_attn_varlen_func as __flash_attn_unpadded_func
            else:
                from flash_attn.flash_attn_interface import flash_attn_unpadded_func as __flash_attn_unpadded_func
        flash_attn_unpadded_func = __flash_attn_unpadded_func
        flash_attn_func = _flash_attn_func
    except ImportError:
        logger.warn(
            "Warning: import flash_attn fail, please install FlashAttention to get higher efficiency "
            "https://github.com/Dao-AILab/flash-attention"
        )

def quantize_cache_v(fdata, bits, qmax, qmin):
    # b, s, head, h-dim->b, head, s, h-dim
    qtype = torch.uint8
    device = fdata.device
    shape = fdata.shape

    fdata_cal = torch.flatten(fdata, 2)
    fmax = torch.amax(fdata_cal, dim=-1, keepdim=True)
    fmin = torch.amin(fdata_cal, dim=-1, keepdim=True)
    # Compute params
    if qmax.device != fmax.device:
        qmax = qmax.to(device)
        qmin = qmin.to(device)
    scale = (fmax - fmin) / (qmax - qmin)
    zero = qmin - fmin / scale
    scale = scale.unsqueeze(-1).repeat(1,1,shape[2],1).contiguous()
    zero = zero.unsqueeze(-1).repeat(1,1,shape[2],1).contiguous()
    # Quantize
    res_data = fdata / scale + zero
    qdata = torch.clamp(res_data, qmin, qmax).to(qtype)
    return qdata.contiguous(), scale, zero

def dequantize_cache_torch(qdata, scale, zero):
    data = scale * (qdata - zero)
    return data

class FlashSelfAttention(torch.nn.Module):
    def __init__(
        self,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
    ):
        super().__init__()
        assert flash_attn_unpadded_func is not None, (
            "Please install FlashAttention first, " "e.g., with pip install flash-attn"
        )
        assert (
            rearrange is not None
        ), "Please install einops first, e.g., with pip install einops"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def unpad_input(self, hidden_states, attention_mask):
        valid_mask = attention_mask.squeeze(1).squeeze(1).eq(0)
        seqlens_in_batch = valid_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(valid_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
        hidden_states = hidden_states[indices]
        return hidden_states, indices, cu_seqlens, max_seqlen_in_batch

    def pad_input(self, hidden_states, indices, batch, seqlen):
        output = torch.zeros(batch * seqlen, *hidden_states.shape[1:], device=hidden_states.device,
                             dtype=hidden_states.dtype)
        output[indices] = hidden_states
        return rearrange(output, '(b s) ... -> b s ...', b=batch)

    def forward(self, q, k, v, attention_mask=None):
        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
        assert all((i.is_cuda for i in (q, k, v)))
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        seqlen_out = seqlen_q

        if flash_attn_func is not None and batch_size == 1:
            dropout_p = self.dropout_p if self.training else 0
            output = flash_attn_func(q, k, v, dropout_p, softmax_scale=self.softmax_scale, causal=self.causal)
            return output

        q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q.device,
        )

        if batch_size > 1 and attention_mask is not None:
            k, indices_k, cu_seqlens_k, seqlen_k = self.unpad_input(k, attention_mask)
            if q.size(0) == v.size(0):
                q = q[indices_k]
                cu_seqlens_q = cu_seqlens_k
                seqlen_q = seqlen_k
            v = v[indices_k]
        else:
            cu_seqlens_k = torch.arange(
                0,
                (batch_size + 1) * seqlen_k,
                step=seqlen_k,
                dtype=torch.int32,
                device=q.device,
            )

        if self.training:
            assert seqlen_k == seqlen_q
            is_causal = self.causal
            dropout_p = self.dropout_p
        else:
            is_causal = seqlen_q == seqlen_k
            dropout_p = 0

        output = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale,
            causal=is_causal,
        )
        if batch_size > 1 and attention_mask is not None and seqlen_q == seqlen_k:
            output = self.pad_input(output, indices_k, batch_size, seqlen_out)
        else:
            new_shape = (batch_size, output.shape[0] // batch_size) + output.shape[1:]
            output = output.view(new_shape)
        return output


class QWenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        self.seq_length = config.seq_length

        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.use_flash_attn = config.use_flash_attn
        self.scale_attn_weights = True

        self.projection_size = config.kv_channels * config.num_attention_heads

        assert self.projection_size % config.num_attention_heads == 0
        self.hidden_size_per_attention_head = (
            self.projection_size // config.num_attention_heads
        )

        self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)

        self.c_proj = nn.Linear(
            config.hidden_size, self.projection_size, bias=not config.no_bias
        )

        self.is_fp32 = not (config.bf16 or config.fp16)
        if (
            self.use_flash_attn
            and flash_attn_unpadded_func is not None
            and not self.is_fp32
        ):
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=config.attn_dropout_prob
            )
        self.bf16 = config.bf16

        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn

        logn_list = [
            math.log(i, self.seq_length) if i > self.seq_length else 1
            for i in range(1, 32768)
        ]
        logn_tensor = torch.tensor(logn_list)[None, :, None, None]
        self.register_buffer("logn_tensor", logn_tensor, persistent=False)

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.softmax_in_fp32 = config.softmax_in_fp32 if hasattr(config, 'softmax_in_fp32') else False
        self.use_cache_quantization = config.use_cache_quantization if hasattr(config, 'use_cache_quantization') else False
        self.use_cache_kernel = config.use_cache_kernel if hasattr(config,'use_cache_kernel') else False
        cache_dtype = torch.float
        if self.bf16:
            cache_dtype=torch.bfloat16
        elif config.fp16:
            cache_dtype = torch.float16
        self.cache_qmax = torch.tensor(torch.iinfo(torch.uint8).max, dtype=cache_dtype)
        self.cache_qmin = torch.tensor(torch.iinfo(torch.uint8).min, dtype=cache_dtype)

        if config.use_cache_quantization and config.use_cache_kernel:
            # pre check if the support files existing
            module_root = pathlib.Path(__file__).parent
            src_files = ("cache_autogptq_cuda_256.cpp", "cache_autogptq_cuda_kernel_256.cu")
            if any(not (module_root/src).is_file() for src in src_files):
                warnings.warn("KV cache kernel source files (.cpp and .cu) not found.")
                self.cache_kernels = None
            else:
                try:
                    from .cpp_kernels import cache_autogptq_cuda_256
                    self.cache_kernels = cache_autogptq_cuda_256
                except ImportError:
                    warnings.warn("Failed to import KV cache kernels.")
                    self.cache_kernels = None

    def _attn(self, query, key, value, causal_mask=None, attention_mask=None, head_mask=None):
        device = query.device
        if self.use_cache_quantization:
            qk, qk_scale, qk_zero = key
            if self.use_cache_kernel and self.cache_kernels is not None:
                shape = query.shape[:-1] + (qk.shape[-2],)
                attn_weights = torch.zeros(shape, dtype=torch.float16, device=device)
                self.cache_kernels.vecquant8matmul_batched_faster_old(
                    query.contiguous() if query.dtype == torch.float16 else query.to(torch.float16).contiguous(),
                    qk.transpose(-1, -2).contiguous(),
                    attn_weights,
                    qk_scale.contiguous() if qk_scale.dtype == torch.float16 else qk_scale.to(torch.float16).contiguous(),
                    qk_zero.contiguous()if qk_zero.dtype == torch.float16 else qk_zero.to(torch.float16).contiguous())
                # attn_weights = attn_weights.to(query.dtype).contiguous()
            else:
                key = dequantize_cache_torch(qk, qk_scale, qk_zero)
                attn_weights = torch.matmul(query, key.transpose(-1, -2))
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            if self.use_cache_quantization:
                size_temp = value[0].size(-1)
            else:
                size_temp = value.size(-1)
            attn_weights = attn_weights / (size_temp ** 0.5)

        mask_value = torch.finfo(attn_weights.dtype).min
        if causal_mask is not None:
            attn_weights = torch.where(
                causal_mask, attn_weights.to(attn_weights.dtype), mask_value
            )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        if self.softmax_in_fp32:
            attn_weights = nn.functional.softmax(attn_weights.float(), dim=-1)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights.type(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        if self.use_cache_quantization:
            qv, qv_scale, qv_zero = value
            if self.use_cache_kernel and self.cache_kernels is not None:
                shape = attn_weights.shape[:-1] + (query.shape[-1],)
                attn_output = torch.zeros(shape, dtype=torch.float16, device=device)
                self.cache_kernels.vecquant8matmul_batched_column_compression_faster_old(
                    attn_weights.contiguous() if attn_weights.dtype == torch.float16 else attn_weights.to(torch.float16).contiguous(),
                    qv.contiguous(),  # dtype: int32
                    attn_output,
                    qv_scale.contiguous() if qv_scale.dtype == torch.float16 else qv_scale.to(torch.float16).contiguous(),
                    qv_zero.contiguous() if qv_zero.dtype == torch.float16 else qv_zero.to(torch.float16).contiguous())
                if attn_output.dtype != query.dtype:
                    attn_output = attn_output.to(query.dtype)
                    attn_weights = attn_weights.to(query.dtype)
            else:
                value = dequantize_cache_torch(qv, qv_scale, qv_zero)
                attn_output = torch.matmul(attn_weights, value)
        else:
            attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        rotary_pos_emb_list: Optional[List[List[torch.Tensor]]] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        mixed_x_layer = self.c_attn(hidden_states)

        query, key, value = mixed_x_layer.split(self.split_size, dim=2)

#############################################################################################        
        
        # 只加载一次 JSON 文件
        if not hasattr(self, "_sorted_k_indices"):
            with open("/home/lmlu/qwen/k_indices_sorted_by_layer.json", "r") as f:

                self._sorted_k_indices = json.load(f)
        if not hasattr(self, "_sorted_v_indices"):
            with open("/home/lmlu/qwen/v_indices_sorted_by_layer.json", "r") as f:
                self._sorted_v_indices = json.load(f)
        b, s, d = key.shape
        def modify_kv_by_sorted_indices(kv_tensor, sorted_indices_dict, layer_idx, percentage=0.8):
            b, s, d = kv_tensor.shape
            sorted_indices = sorted_indices_dict[str(layer_idx)]
            num_to_modify = int(percentage * len(sorted_indices))
            dims_to_modify = torch.tensor(sorted_indices[:num_to_modify], device=kv_tensor.device)
            even_len = s if s % 2 == 0 else s - 1
            # if(layer_idx == 1):
            #     print(even_len,flush=True)
            # if(s==1):
            #     print(114514)
            t0 = kv_tensor[:, 0:even_len:2, :][:, :, dims_to_modify]
            t1 = kv_tensor[:, 1:even_len:2, :][:, :, dims_to_modify]
            avg = (t0 + t1) / 2

            kv_tensor = kv_tensor.clone()
            kv_tensor[:, 0:even_len:2, :].scatter_(2, dims_to_modify[None, None, :].expand(b, even_len // 2, -1), avg)
            kv_tensor[:, 1:even_len:2, :].scatter_(2, dims_to_modify[None, None, :].expand(b, even_len // 2, -1), avg)
            return kv_tensor

        # 分别处理 key 和 value
        key = modify_kv_by_sorted_indices(key, self._sorted_k_indices, self.layer_idx, percentage=0.1)
        value = modify_kv_by_sorted_indices(value, self._sorted_v_indices, self.layer_idx, percentage=0.1)

        # def group_tokens(tensor, dims, group_size):
        #     """
        #     对指定维度 dims 上的 token，每 group_size 个 token 平均成一样值（矢量化）
        #     tensor: [b, s, d]
        #     dims: [d']
        #     """
        #     b, s, d = tensor.shape
        #     s_g = (s // group_size) * group_size  # 保证整除
        #     tensor = tensor.clone()

        #     views = [tensor[:, i:s_g:group_size, :][:, :, dims] for i in range(group_size)]
        #     avg = sum(views) / group_size

        #     for i in range(group_size):
        #         tensor[:, i:s_g:group_size, :].scatter_(
        #             2, dims[None, None, :].expand(b, s_g // group_size, -1), avg
        #         )
        #     return tensor

        # def modify_key_by_layer_rule(k_tensor, sorted_indices_dict, layer_idx):
        #     b, s, d = k_tensor.shape
        #     sorted_indices = sorted_indices_dict[str(layer_idx)]
        #     total = len(sorted_indices)
        #     i50 = int(0.5 * total)
        #     i70 = int(0.7 * total)

        #     dims_4 = torch.tensor(sorted_indices[:i50], device=k_tensor.device)
        #     dims_2 = torch.tensor(sorted_indices[i50:i70], device=k_tensor.device)

        #     if layer_idx < 16:
        #         #k_tensor = group_tokens(k_tensor, dims_4, group_size=4)
        #         k_tensor = group_tokens(k_tensor, dims_4, group_size=4)
        #         k_tensor = group_tokens(k_tensor, dims_2, group_size=2)
        #     else:
        #         k_tensor = group_tokens(k_tensor, dims_4, group_size=2)
        #     return k_tensor

        # def modify_value_by_layer_rule(v_tensor, sorted_indices_dict, layer_idx):
        #     if layer_idx >= 16:
        #         return v_tensor
        #     b, s, d = v_tensor.shape
        #     sorted_indices = sorted_indices_dict[str(layer_idx)]
        #     i50 = int(0.5 * len(sorted_indices))
        #     dims_2 = torch.tensor(sorted_indices[:i50], device=v_tensor.device)

        #     return group_tokens(v_tensor, dims_2, group_size=2)

        # key = modify_key_by_layer_rule(key, self._sorted_k_indices, self.layer_idx)
        # value = modify_value_by_layer_rule(value, self._sorted_v_indices, self.layer_idx)

        
 ##############################################################################################       
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if rotary_pos_emb_list is not None:
            cur_len = query.shape[1]
            if len(rotary_pos_emb_list) == 1:
                rotary_pos_emb = rotary_pos_emb_list[0]
                rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
                rotary_pos_emb = (rotary_pos_emb,) * 2
                q_pos_emb, k_pos_emb = rotary_pos_emb
                # Slice the pos emb for current inference
                query = apply_rotary_pos_emb(query, q_pos_emb)
                key = apply_rotary_pos_emb(key, k_pos_emb)
            else:
                query_list = []
                key_list = []
                for i, rotary_pos_emb in enumerate(rotary_pos_emb_list):
                    rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
                    rotary_pos_emb = (rotary_pos_emb,) * 2
                    q_pos_emb, k_pos_emb = rotary_pos_emb
                    # Slice the pos emb for current inference
                    query_list += [apply_rotary_pos_emb(query[i:i+1, :, :], q_pos_emb)]
                    key_list += [apply_rotary_pos_emb(key[i:i+1, :, :], k_pos_emb)]
                query = torch.cat(query_list, dim=0)
                key = torch.cat(key_list, dim=0)

        if self.use_cache_quantization:
            key = quantize_cache_v(key.permute(0, 2, 1, 3),
                                       bits=8,
                                       qmin=self.cache_qmin,
                                       qmax=self.cache_qmax)
            value = quantize_cache_v(value.permute(0, 2, 1, 3),
                                         bits=8,
                                         qmin=self.cache_qmin,
                                         qmax=self.cache_qmax)


        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            if self.use_cache_quantization:
                # use_cache_quantization:
                # present=((q_key,key_scale,key_zero_point),
                #          (q_value,value_scale,value_zero_point))
                key = (torch.cat((past_key[0], key[0]), dim=2),
                       torch.cat((past_key[1], key[1]), dim=2),
                       torch.cat((past_key[2], key[2]), dim=2))
                value = (torch.cat((past_value[0], value[0]), dim=2),
                         torch.cat((past_value[1], value[1]), dim=2),
                         torch.cat((past_value[2], value[2]), dim=2))
            else:
                # not use_cache_quantization:
                # present=(key,value)
                key = torch.cat((past_key, key), dim=1)
                value = torch.cat((past_value, value), dim=1)

#####################################################################################################
        #提取RoPE之后的kv
        
        # save_dir = "/home/lmlu/qwen/qwen_norm_diff"
        # os.makedirs(save_dir, exist_ok=True)
        # if (s>1):
        #     def process_and_save(tensor, name):
        #         # 差分并归一化：[B, S-1, H, D]
        #         diff = tensor[:, 1:, :, :] - tensor[:, :-1, :, :]
        #         diff = diff.abs() / (diff.norm(dim=-1, keepdim=True) + 1e-6)

        #         # 平均化序列维度 -> [B, H, D]
        #         diff_mean = diff.mean(dim=1)

        #         # 转为 numpy 数组
        #         diff_np = diff_mean.cpu().float().numpy()  # [B, H, D]

        #         # 文件路径
        #         path = os.path.join(save_dir, f"diffvec_{name}_layer{self.layer_idx}.npy")

        #         # 追加保存
        #         if os.path.exists(path):
        #             prev = np.load(path, allow_pickle=True)
        #             out = np.concatenate([prev, diff_np], axis=0)
        #         else:
        #             out = diff_np

        #         np.save(path, out)

        #     # 分别处理 key 和 value
        #     process_and_save(key, "key")
        #     process_and_save(value, "value")
        ###############################
        # if not hasattr(self, "_sorted_indices_cache"):
        #     self._sorted_indices_cache = {
        #         "key": json.load(open("/home/lmlu/qwen/qwen_diff_key_all_sort_index.json")),
        #         "value": json.load(open("/home/lmlu/qwen/qwen_diff_value_all_sort_index.json"))
        #     }
        # def smooth_low_dims_vectorized_cached(tensor, layer_idx, sort_index_cache, dim_ratio=0.6):
        #     """
        #     对张量的低变化维度进行成对平滑，支持奇数长度序列。
        #     使用预缓存的排序索引，避免重复读取 JSON。
            
        #     Args:
        #         tensor: [B, S, H, D] 输入张量
        #         layer_idx: 当前层索引
        #         sort_index_cache: dict，包含预加载的排序索引
        #         dim_ratio: 参与平滑的维度比例（默认前 20% 的维度）
                
        #     Returns:
        #         [B, S, H, D]，平滑处理后的张量
        #     """
        #     key = f"layer_{layer_idx}"
        #     if key not in sort_index_cache:
        #         print(f"Warning: Layer {layer_idx} not in sort index.")
        #         return tensor

        #     B, S, H, D = tensor.shape
        #     tensor = tensor.clone()
        #     S_even = S - (S % 2)

        #     # 前 S_even 个 token 做配对
        #     paired = tensor[:, :S_even].reshape(B, S_even // 2, 2, H, D)
        #     K = int(D * dim_ratio)

        #     # 读取所有 head 的前 K 个排序维度索引
        #     sort_idx_tensor = torch.tensor(
        #         [sort_index_cache[key][f"head_{h}"][:K] for h in range(H)],
        #         device=tensor.device
        #     )  # [H, K]

        #     # 扩展索引到 [B, S/2, 2, H, K] 形状以做 gather
        #     sort_idx_expanded = sort_idx_tensor.view(1, 1, 1, H, K).expand(B, S_even // 2, 2, H, K)

        #     # 选出对应维度的值，计算平均后 scatter 回去
        #     selected = torch.gather(paired, dim=-1, index=sort_idx_expanded)
        #     avg = selected.mean(dim=2, keepdim=True).expand(-1, -1, 2, -1, -1)
        #     paired.scatter_(dim=-1, index=sort_idx_expanded, src=avg)

        #     # 重新 reshape 回 [B, S_even, H, D]
        #     smoothed_even = paired.reshape(B, S_even, H, D)

        #     # 如果 S 是奇数，拼接最后一个原始 token
        #     if S % 2 == 1:
        #         last_token = tensor[:, -1:].clone()  # [B, 1, H, D]
        #         tensor = torch.cat([smoothed_even, last_token], dim=1)
        #     else:
        #         tensor = smoothed_even

        #     return tensor

        # key_sort_index = self._sorted_indices_cache["key"]
        # value_sort_index = self._sorted_indices_cache["value"]
        # key = smooth_low_dims_vectorized_cached(key, self.layer_idx, key_sort_index)
        # value = smooth_low_dims_vectorized_cached(value, self.layer_idx, value_sort_index)
        ##########################
        # def average_token_pairs(tensor):
        #     """
        #     对 tensor 的相邻 token 对做均值处理（每两个 token 共享均值）。
        #     tensor: [B, S, H, D]
        #     return: [B, S, H, D]
        #     """
        #     B, S, H, D = tensor.shape
        #     S_even = S - (S % 2)  # 取偶数部分，避免尾部不完整对
        #     paired = (tensor[:, 0:S_even:2] + tensor[:, 1:S_even:2]) / 2  # [B, S//2, H, D]
        #     paired = paired.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # 变成 [B, S//2, 2, H, D]
        #     smoothed = paired.reshape(B, S_even, H, D).clone()  # 展平为 [B, S_even, H, D]

        #     if S % 2 == 1:
        #         # 拼接上最后一个 token（不处理）
        #         last = tensor[:, -1:, :, :]
        #         smoothed = torch.cat([smoothed, last], dim=1)

        #     return smoothed
        # key = average_token_pairs(key)
        # value = average_token_pairs(value)

        ###########################
        #直接扔掉一半
        # def discard_odd_tokens(tensor):
        #     """
        #     保留偶数位置 token（即 token[0], token[2], ...）。
        #     tensor: [B, S, H, D]
        #     return: [B, S//2, H, D]
        #     """
        #     return tensor[:, ::2, :, :].contiguous()
        # key = discard_odd_tokens(key)
        # value = discard_odd_tokens(value)
        # # 修剪 attention_mask 的 key 部分（即最后一个维度）
        # # attn_mask shape: [B, 1, S_q, S_k]
        # if attention_mask is not None:
        #     attention_mask = attention_mask[..., ::2]  # 保留 key 的偶数位置

        #######################
        #奇数位用偶数位代替
        # key, value shape: [B, S, H, D]，S可能是奇数
        # key = key.clone()
        # value = value.clone()

        # seq_len = key.size(1)

        # # 获取奇偶位切片并对齐最短长度
        # odd_indices = slice(1, seq_len, 2)  # 偶数下标位置（1, 3, 5, ...）
        # even_indices = slice(0, seq_len - 1, 2)  # 与之对应的前一个位置（0, 2, 4, ...）

        # # 进行替换
        # key[:, odd_indices, :, :] = key[:, even_indices, :, :]
        # value[:, odd_indices, :, :] = value[:, even_indices, :, :]
#####################################################################################################



        if use_cache:
            present = (key, value)
        else:
            present = None

        key_size = key[0].size(2) if self.use_cache_quantization else key.size(1)
        if key_size > self.seq_length and self.use_logn_attn and not self.training:
            if self.use_cache_quantization:
                seq_start = key[0].size(2) - query.size(1)
                seq_end = key[0].size(2)
            else:
                seq_start = key.size(1) - query.size(1)
                seq_end = key.size(1)
            logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :].type_as(query)
            query = query * logn_tensor.expand_as(query)

        if (
            self.use_flash_attn
            and flash_attn_unpadded_func is not None
            and not self.is_fp32
            and query.is_cuda
        ):
            q, k, v = query, key, value
            attn_output = self.core_attention_flash(q, k, v, attention_mask=attention_mask)
        else:
            key_size = key[0].size(2) if self.use_cache_quantization else key.size(1)
            if query.size(1) == key_size:
                causal_mask = torch.tril(
                    torch.ones((key_size, key_size), dtype=torch.bool, device=query.device)
                ).view(1, 1, key_size, key_size)
            else:
                causal_mask = None
            query = query.permute(0, 2, 1, 3)
            if not self.use_cache_quantization:
                key = key.permute(0, 2, 1, 3)
                value = value.permute(0, 2, 1, 3)
            if (
                causal_mask is None
                and self.use_flash_attn
                and flash_attn_unpadded_func is not None
                and not self.is_fp32
                and not query.is_cuda
            ):
                raise Exception(_ERROR_INPUT_CPU_QUERY_WITH_FLASH_ATTN_ACTIVATED)

            if not self.use_cache_quantization and SUPPORT_TORCH2:
                if attention_mask is not None:
                    attention_mask = attention_mask.expand(-1, -1, query.size(2), -1)
                    if causal_mask is not None:
                        attention_mask = attention_mask.masked_fill(~causal_mask, torch.finfo(query.dtype).min)
                else:
                    attention_mask = causal_mask
                attn_output = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask
                ).transpose(1, 2)
                attn_weight = None
            else:
                attn_output, attn_weight = self._attn(
                    query, key, value, causal_mask, attention_mask, head_mask
                )
        context_layer = self._merge_heads(
            attn_output, self.num_heads, self.head_dim
        )

        attn_output = self.c_proj(context_layer)

        outputs = (attn_output, present)
        if output_attentions:
            if (
                self.use_flash_attn
                and flash_attn_unpadded_func is not None
                and not self.is_fp32
            ):
                raise ValueError("Cannot output attentions while using flash-attn")
            elif not self.use_cache_quantization and SUPPORT_TORCH2:
                raise ValueError("Cannot output attentions while using scaled_dot_product_attention")
            else:
                outputs += (attn_weight,)

        return outputs


class QWenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias
        )
        self.w2 = nn.Linear(
            config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias
        )
        ff_dim_in = config.intermediate_size // 2
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias=not config.no_bias)

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output


class QWenBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        hidden_size = config.hidden_size
        self.bf16 = config.bf16

        self.ln_1 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.attn = QWenAttention(config)
        self.attn.layer_idx = layer_idx  # ✅ 关键注入
        self.ln_2 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.mlp = QWenMLP(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        rotary_pos_emb_list: Optional[List[List[torch.Tensor]]] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        layernorm_output = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            layernorm_output,
            rotary_pos_emb_list,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        residual = hidden_states
        layernorm_input = attn_output + residual

        layernorm_output = self.ln_2(layernorm_input)

        residual = layernorm_input
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class QWenPreTrainedModel(PreTrainedModel):
    config_class = QWenConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["QWenBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                p.data.normal_(
                    mean=0.0,
                    std=(
                        self.config.initializer_range
                        / math.sqrt(2 * self.config.num_hidden_layers)
                    ),
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, QWenModel):
            module.gradient_checkpointing = value


class QWenModel(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.use_cache_quantization = self.config.use_cache_quantization if hasattr(self.config, 'use_cache_quantization') else False

        self.gradient_checkpointing = False
        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.seq_length = config.seq_length

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)

        self.drop = nn.Dropout(config.emb_dropout_prob)

        if config.rotary_pct == 1.0:
            self.rotary_ndims = None
        else:
            assert config.rotary_pct < 1
            self.rotary_ndims = int(
                config.kv_channels * config.rotary_pct
            )
        dim = (
            self.rotary_ndims
            if self.rotary_ndims is not None
            else config.kv_channels
        )
        self.rotary_emb = RotaryEmbedding(dim, base=config.rotary_emb_base)

        self.use_flash_attn = config.use_flash_attn
        self.is_fp32 = not (config.bf16 or config.fp16)

        self.h = nn.ModuleList(
            [
                QWenBlock(
                    config,
                    layer_idx = i
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_f = RMSNorm(
            self.embed_dim,
            eps=config.layer_norm_epsilon,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def get_ntk_alpha(self, true_seq_len):
        context_value = math.log(true_seq_len / self.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            if self.use_cache_quantization:
                past_length = past_key_values[0][0][0].size(2)
            else:
                past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds

        kv_seq_len = hidden_states.size()[1]
        if past_key_values[0] is not None:
            # past key values[0][0] shape: bs * seq_len * head_num * dim
            if self.use_cache_quantization:
                kv_seq_len += past_key_values[0][0][0].shape[2]
            else:
                kv_seq_len += past_key_values[0][0].shape[1]

        if self.training or not self.use_dynamic_ntk:
            ntk_alpha_list = [1.0]
        elif kv_seq_len != hidden_states.size()[1]:
            ntk_alpha_list = self.rotary_emb._ntk_alpha_cached_list
        else:
            ntk_alpha_list = []
            if attention_mask is not None and kv_seq_len > self.seq_length:
                true_seq_lens = attention_mask.squeeze(1).squeeze(1).eq(0).sum(dim=-1, dtype=torch.int32)
                for i in range(hidden_states.size()[0]):
                    true_seq_len = true_seq_lens[i].item()
                    ntk_alpha = self.get_ntk_alpha(true_seq_len)
                    ntk_alpha_list.append(ntk_alpha)
            else:
                ntk_alpha = self.get_ntk_alpha(kv_seq_len)
                ntk_alpha_list.append(ntk_alpha)
        self.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
        rotary_pos_emb_list = [
            self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha) for ntk_alpha in ntk_alpha_list
        ]

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    rotary_pos_emb_list,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    rotary_pos_emb_list=rotary_pos_emb_list,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class QWenLMHeadModel(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.masked_bias"]

    def __init__(self, config):
        super().__init__(config)
        assert (
            config.bf16 + config.fp16 + config.fp32 <= 1
        ), "Only one of \"bf16\", \"fp16\", \"fp32\" can be true"

        autoset_precision = config.bf16 + config.fp16 + config.fp32 == 0

        if autoset_precision:
            if SUPPORT_BF16:
                logger.warn(
                    "The model is automatically converting to bf16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.bf16 = True
            elif SUPPORT_FP16:
                logger.warn(
                    "The model is automatically converting to fp16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.fp16 = True
            else:
                config.fp32 = True

        if config.bf16 and SUPPORT_CUDA and not SUPPORT_BF16:
            logger.warn("Your device does NOT seem to support bf16, you can switch to fp16 or fp32 by by passing fp16/fp32=True in \"AutoModelForCausalLM.from_pretrained\".")
        if config.fp16 and SUPPORT_CUDA and not SUPPORT_FP16:
            logger.warn("Your device does NOT support faster inference with fp16, please switch to fp32 which is likely to be faster")
        if config.fp32:
            if SUPPORT_BF16:
                logger.warn("Your device support faster inference by passing bf16=True in \"AutoModelForCausalLM.from_pretrained\".")
            elif SUPPORT_FP16:
                logger.warn("Your device support faster inference by passing fp16=True in \"AutoModelForCausalLM.from_pretrained\".")

        if config.use_flash_attn == "auto":
            if config.bf16 or config.fp16:
                logger.warn("Try importing flash-attention for faster inference...")
                config.use_flash_attn = True
            else:
                config.use_flash_attn = False
        if config.use_flash_attn and config.fp32:
            logger.warn("Flash attention will be disabled because it does NOT support fp32.")

        if config.use_flash_attn:
            _import_flash_attn()

        self.transformer = QWenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.bf16:
            self.transformer.bfloat16()
            self.lm_head.bfloat16()
        if config.fp16:
            self.transformer.half()
            self.lm_head.half()
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if input_ids.size(0) == 1:
            attention_mask = None
        else:
            attention_mask = kwargs.get("attention_mask", None)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:

        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )

    def chat(
        self,
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: Optional[HistoryType],
        system: str = "You are a helpful assistant.",
        stream: Optional[bool] = _SENTINEL,
        stop_words_ids: Optional[List[List[int]]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Tuple[str, HistoryType]:
        generation_config = generation_config if generation_config is not None else self.generation_config

        assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
        if history is None:
            history = []
        else:
            # make a copy of the user's input such that is is left untouched
            history = copy.deepcopy(history)

        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))
        input_ids = torch.tensor([context_tokens]).to(self.device)
        outputs = self.generate(
                    input_ids,
                    stop_words_ids=stop_words_ids,
                    return_dict_in_generate=False,
                    generation_config=generation_config,
                    **kwargs,
                )

        response = decode_tokens(
            outputs[0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=generation_config.chat_format,
            verbose=False,
            errors='replace'
        )

        # as history is a copy of the user inputs,
        # we can always return the new turn to the user.
        # separating input history and output history also enables the user
        # to implement more complex history management
        history.append((query, response))

        return response, history

    def chat_stream(
            self,
            tokenizer: PreTrainedTokenizer,
            query: str,
            history: Optional[HistoryType],
            system: str = "You are a helpful assistant.",
            stop_words_ids: Optional[List[List[int]]] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            generation_config: Optional[GenerationConfig] = None,
            **kwargs,
    ) -> Generator[str, Any, None]:
        generation_config = generation_config if generation_config is not None else self.generation_config
        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT
        if history is None:
            history = []
        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))
        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)
        input_ids = torch.tensor([context_tokens]).to(self.device)

        from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
        self.__class__.generate_stream = NewGenerationMixin.generate
        self.__class__.sample_stream = NewGenerationMixin.sample_stream
        stream_config = StreamGenerationConfig(**generation_config.to_dict(), do_stream=True)

        def stream_generator():
            outputs = []
            for token in self.generate_stream(
                    input_ids,
                    return_dict_in_generate=False,
                    generation_config=stream_config,
                    logits_processor=logits_processor,
                    seed=-1,
                    **kwargs):
                outputs.append(token.item())
                yield tokenizer.decode(outputs, skip_special_tokens=True, errors='ignore')

        return stream_generator()

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generation_config = generation_config if generation_config is not None else self.generation_config

        # Process stop_words_ids.
        stop_words_ids = kwargs.pop("stop_words_ids", None)
        if stop_words_ids is None and generation_config is not None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)
        if stop_words_ids is None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)

        return super().generate(
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **kwargs,
        )


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if importlib.util.find_spec("einops") is None:
            raise RuntimeError("einops is required for Rotary Embedding")

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0
        self._ntk_alpha_cached_list = [1.0]

    def update_rotary_pos_emb_cache(self, seqlen, ntk_alpha=1.0):
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()
                    / self.dim
                )
            )
            self._seq_len_cached = max(2 * seqlen, 16)
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(self._seq_len_cached, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1)
            from einops import rearrange

            emb = rearrange(emb, "n d -> 1 n 1 d")

            cos, sin = emb.cos(), emb.sin()
            self._rotary_pos_emb_cache = [cos, sin]

    def forward(self, max_seq_len, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, ntk_alpha)
        cos, sin = self._rotary_pos_emb_cache
        return [cos[:, :max_seq_len], sin[:, :max_seq_len]]


def _rotate_half(x):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """ Apply rotary embedding to the first rotary_dim of the iput

    Arguments:
      t (tensor(batch_size, seq_len, n_head, head_dim)):
        the input embedding/hidden states
      freqs (list[tensor(1, seq_len, 1, rotary_dim), tensor(1, seq_len, 1, rotary_dim)]):
        the cached cos/sin position embeddings
    """
    rot_dim = freqs[0].shape[-1]
    cos, sin = freqs
    t_float = t.float()
    if apply_rotary_emb_func is not None and t.is_cuda:
        # apply_rotary_emb in flash_attn requires cos/sin to be of
        # shape (seqlen, rotary_dim / 2) and apply rotary embedding
        # to the first rotary_dim of the input
        cos = cos.squeeze(0).squeeze(1)[:, : rot_dim // 2]
        sin = sin.squeeze(0).squeeze(1)[:, : rot_dim // 2]
        return apply_rotary_emb_func(t_float, cos, sin).type_as(t)
    else:
        t_rot, t_pass = t_float[..., :rot_dim], t_float[..., rot_dim:]
        t_rot = (t_rot * cos) + (_rotate_half(t_rot) * sin)
        return torch.cat((t_rot, t_pass), dim=-1).type_as(t)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if rms_norm is not None and x.is_cuda:
            return rms_norm(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
