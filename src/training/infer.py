# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from aigc_zoo.model_zoo.llm.llm_model import MyTransformer
from aigc_zoo.utils.llm_generate import Generate
from aigc_zoo.model_zoo.llm.llm_model import RotaryNtkScaledArguments,RotaryLinearScaledArguments # aigc-zoo 0.1.20

deep_config = get_deepspeed_config()


def infer_tiger(model,tokenizer,max_input_length=512):
    tok_ins = "\n\n### Instruction:\n"
    tok_res = "\n\n### Response:\n"
    prompt_input = tok_ins + "{instruction}" + tok_res

    generation_config = {
        "do_sample": True,
        "eos_token_id": 2,
        "max_length": max_input_length,
        "pad_token_id": 60514,
        "repetition_penalty": 1.1,
        "temperature": 0.3,
        "transformers_version": "4.31.0"
    }
    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 ]

    for input in text_list:
        sess_text = ''

        query_text = input.strip()
        sess_text += tok_ins + query_text
        input_text = prompt_input.format_map({'instruction': sess_text.split(tok_ins, 1)[1]})
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=max_input_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, **generation_config)
        output_str = tokenizer.decode(output[0], skip_special_tokens=False, spaces_between_special_tokens=False)
        answer = output_str.rsplit(tok_res, 1)[1].strip()
        if answer.endswith(tokenizer.eos_token):
            answer = answer.rsplit(tokenizer.eos_token, 1)[0].strip()

        print('input', input)
        print('output', answer)

if __name__ == '__main__':


    parser = HfArgumentParser((ModelArguments,))
    (model_args,)  = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config()

    enable_ntk = False
    rope_args = None
    if enable_ntk and config.model_type == 'llama':
        rope_args = RotaryNtkScaledArguments(name='rotary_emb',max_position_embeddings=2048, alpha=4)  # 扩展 8k
        # rope_args = RotaryLinearScaledArguments(name='rotary_emb',max_position_embeddings=2048, scale=4) # 扩展 8k


    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype,rope_args=rope_args)
    model = pl_model.get_llm_model()
    model = model.eval()
    if hasattr(model,'quantize'):
        # 支持llama llama2量化
        if not model.quantized:
            # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
            model.half().quantize(4).cuda()
            # 保存量化权重
            # model.save_pretrained('llama2-7b-chat-int4',max_shard_size="2GB")
            # exit(0)
        else:
            # 已经量化
            model.half().cuda()
    else:
        model.half().cuda()

    if train_info_args['model_name_or_path'].lower().find('tiger') >=0:
        infer_tiger(model,tokenizer)
    else:
        text_list = ["写一个诗歌，关于冬天",
                     "晚上睡不着应该怎么办",
                     "从南京到上海的路线",
                     ]
        for input in text_list:
            response = Generate.generate(model, query=input, tokenizer=tokenizer, max_length=512,
                                              eos_token_id=config.eos_token_id,
                                              do_sample=False, top_p=0.7, temperature=0.95, )
            print('input', input)
            print('output', response)