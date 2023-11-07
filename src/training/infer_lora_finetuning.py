# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer_lora_finetuning
import os
import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser,AutoConfig
from data_utils import train_info_args, NN_DataHelper,global_args,build_template
from aigc_zoo.model_zoo.llm.llm_model import MyTransformer,PetlArguments,PromptArguments
from aigc_zoo.utils.llm_generate import Generate


if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(train_info_args, allow_extra_keys=True)


    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config()
    

    # 一般根据时间排序选最新的权重文件夹
    ckpt_dir = './best_ckpt/last'

    config = AutoConfig.from_pretrained(ckpt_dir)
    lora_args = PetlArguments.from_pretrained(ckpt_dir)

    assert lora_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size',None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args,
                             lora_args=lora_args,
                             torch_dtype=config.torch_dtype,
                             new_num_tokens=new_num_tokens,
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )

    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir)

    pl_model.eval().half().cuda()

    enable_merge_weight = False

    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_sft_weight(os.path.join(ckpt_dir, 'pytorch_model_merge.bin'), merge_lora_weight=True)
    else:
        model = pl_model.get_llm_model()

        text_list = ["写一个诗歌，关于冬天",
                     "晚上睡不着应该怎么办",
                     "从南京到上海的路线",
                     ]
        for input in text_list:
            response = Generate.generate(model, query=build_template(input), tokenizer=tokenizer, max_length=512,
                                         eos_token_id=config.eos_token_id,
                                         do_sample=False, top_p=0.7, temperature=0.95, )
            print('input', input)
            print('output', response)