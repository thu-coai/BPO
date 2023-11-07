# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps

__model_path__ = {
    'bloom-560m': {
        'model_type': 'bloom',
        'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-560m',
        'config_name': '/data/nlp/pre_models/torch/bloom/bloom-560m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-560m',
    },
    'bloom-1b7': {
        'model_type': 'bloom',
        'model_name_or_path': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
        'config_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/bloom/bloom-1b7',
    },
    'opt-350m': {
        'model_type': 'opt',
        'model_name_or_path': '/data/nlp/pre_models/torch/opt/opt-350m',
        'config_name': '/data/nlp/pre_models/torch/opt/opt-350m/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/opt/opt-350m',
    },

    'llama-7b-hf': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
        'config_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/llama-7b-hf',
    },

    'llama-13b-hf': {
        'model_type': 'llama',
        'model_name_or_path': '/cjl/pretrained_models/llama-13b-hf',
        'config_name': '/cjl/pretrained_models/llama-13b-hf/config.json',
        'tokenizer_name': '/cjl/pretrained_models/llama-13b-hf',
    },

    # TODO change model path
    'Llama-2-7b-chat-hf':{
        'model_type': 'llama',
        'model_name_or_path': '/cjl/pretrained_models/Llama-2-7b-chat-hf',
        'config_name': '/cjl/pretrained_models/Llama-2-7b-chat-hf/config.json',
        'tokenizer_name': '/cjl/pretrained_models/Llama-2-7b-chat-hf',
    },

    'Llama2-Chinese-7b-Chat':{
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-7b-Chat',
        'config_name': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-7b-Chat/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-7b-Chat',
    },

    'Llama2-Chinese-13b-Chat':{
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-13b-Chat',
        'config_name': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-13b-Chat/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/Llama2-Chinese-13b-Chat',
    },

    'chatyuan-7b': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/ChatYuan-7B',
        'config_name': '/data/nlp/pre_models/torch/llama/ChatYuan-7B/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/ChatYuan-7B',
    },
    'tigerbot-13b-chat': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat',
        'config_name': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat',
    },
    'tigerbot-13b-chat-int4': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat-int4',
        'config_name': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/tigerbot-13b-chat-int4',
    },

    'openbuddy-llama2-70b-v10.1': {
        'model_type': 'llama',
        'model_name_or_path': '/data/nlp/pre_models/torch/llama/openbuddy-llama2-70b-v10.1-bf16',
        'config_name': '/data/nlp/pre_models/torch/llama/openbuddy-llama2-70b-v10.1-bf16/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/llama/openbuddy-llama2-70b-v10.1-bf16',
    },



    'rwkv-4-430m-pile': {
        'model_type': 'rwkv',
        'model_name_or_path': '/data/nlp/pre_models/torch/rwkv/rwkv-4-430m-pile',
        'config_name': '/data/nlp/pre_models/torch/rwkv/rwkv-4-430m-pile/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/rwkv/rwkv-4-430m-pile',
    },

}


# 'target_modules': ['query_key_value'],  # bloom,gpt_neox
# 'target_modules': ["q_proj", "v_proj"], #llama,opt,gptj,gpt_neo
# 'target_modules': ['c_attn'], #gpt2
# 'target_modules': ['project_q','project_v'] # cpmant

train_target_modules_maps = {
    't5': ['qkv_proj'],
    'moss': ['qkv_proj'],
    'chatglm': ['query_key_value'],
    'bloom' : ['query_key_value'],
    'gpt_neox' : ['query_key_value'],
    'llama' : ["q_proj", "v_proj"],
    'opt' : ["q_proj", "v_proj"],
    'gptj' : ["q_proj", "v_proj"],
    'gpt_neo' : ["q_proj", "v_proj"],
    'gpt2' : ['c_attn'],
    'cpmant' : ['project_q','project_v'],
    'rwkv' : ['key','value','receptance'],
}


train_model_config = __model_path__['Llama-2-7b-chat-hf']

