import json
import argparse


def cal_overall(input_file, judge_key):
    with open(input_file) as f:
        l = f.readlines()
    w_l_t = [0, 0, 0]
    num = 0

    for i in l:
        i = json.loads(i)
        if "[[A]]" in i['response'].split('\n\n')[-1]:
            if judge_key in i['option_a']:
                num += 1
                w_l_t[1] += 1
            else:
                w_l_t[0] += 1
        elif "[[B]]" in i['response'].split('\n\n')[-1]:
            if judge_key in i['option_a']:
                num += 1
                w_l_t[0] += 1
            else:
                w_l_t[1] += 1
        elif "[[C]]" in i['response'].split('\n\n')[-1]:
            if judge_key in i['option_a']:
                num += 1
            w_l_t[2] += 1

    print(w_l_t)
    print(f"Origin v.s. {judge_key}, win lose tie: ", [i/len(l) for i in w_l_t])
    print(f"{judge_key} as first: ", num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()  

    # TODO there should be a special key in the dict to distinguish the source model, like 'optimized_prompt' will be in the optimized version
    judge_key = 'optimized_prompt'
    cal_overall(args.input_file, judge_key)