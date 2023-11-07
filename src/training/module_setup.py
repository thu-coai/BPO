# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/8/16 16:03

from deep_training.utils.hf import register_transformer_model,register_transformer_config
from transformers import AutoModelForCausalLM
from deep_training.nlp.models.rellama.modeling_llama import LlamaForCausalLM
__all__ = [
    "module_setup"
]

def module_setup():
    # 导入模型
    #register_transformer_config(XverseConfig)
    register_transformer_model(LlamaForCausalLM, AutoModelForCausalLM)