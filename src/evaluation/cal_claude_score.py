import json
import argparse


def cal_overall(input_file, judge_key):
    with open(input_file) as f:
        l = f.readlines()
    w_l_t = [0, 0, 0]
    num = 0

    str_a = "model_1"
    str_b = "model_2"

    print(len(l))
    for i in l:
        i = json.loads(i)
        if i['response'].split('rank')[0].count(str_a):
            if judge_key in i['option_a']:
                num += 1
                w_l_t[1] += 1
            else:
                w_l_t[0] += 1
        elif i['response'].split('rank')[0].count(str_b):
            if judge_key in i['option_a']:
                num += 1
                w_l_t[0] += 1
            else:
                w_l_t[1] += 1
        else:
            print(i['response'].split('rank')[0])
    print(w_l_t)
    print(f"Origin v.s. {judge_key}, win lose tie: ", [i / len(l) for i in w_l_t])
    print(f"{judge_key} as first: ", num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()  

    # TODO there should be a special key in the dict to distinguish the source model, like 'optimized_prompt' will be in the optimized version
    judge_key = 'optimized_prompt'
    cal_overall(args.input_file, judge_key)