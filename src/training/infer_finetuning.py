# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, AutoConfig
from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config,build_template
from aigc_zoo.model_zoo.llm.llm_model import MyTransformer
from aigc_zoo.utils.llm_generate import Generate
import json
from tqdm import tqdm

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    # TODO add input file and output file (requires json)
    # or you could implement it yourself
    input_file = '../../data/data_construction/examples_ctx.json'
    output_file = '../../data/data_construction/examples_ctx_optimized_gen.json'
    
    # optimized on evaluation set
    # input_file = '../../data/testset/dolly_eval.json'
    # output_file = '../../data/testset/dolly_eval_optimized.json'

    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _,_= dataHelper.load_tokenizer_and_config()
    

    config = AutoConfig.from_pretrained('./output/best_ckpt')
    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=config.torch_dtype,)

    # deepspeed 权重使用转换脚本命令
    # 一般根据时间排序选最新的权重文件夹
    # cd best_ckpt/last
    # python zero_to_fp32.py . ../last.ckpt

    train_weight = './output/best_ckpt'
    
    pl_model.load_sft_weight(train_weight,strict=True)

    # 保存hf权重
    # config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    # 保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')

    model = pl_model.get_llm_model()

    model.eval().half().cuda()


    with open(input_file, encoding='utf-8') as f:
        text_list = json.load(f)[:]
    
    gen_res = []

    for input in tqdm(text_list[:]):
        
        response = Generate.generate(model, query=build_template((input['instruction']+'\n'+input['context']).strip()), tokenizer=tokenizer, max_new_tokens=1024,
                                     eos_token_id=config.eos_token_id,
                                     do_sample=True, top_p=0.9, temperature=0.6, num_beams=1)

        input['gen_res'] = response.strip()
        gen_res.append(input)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gen_res, f, indent=4, ensure_ascii=False)
