"""OpenkoLLM Leaderboard 경쟁을 위해 Openorca-KO, Platypus-KO 데이터셋을 추가하는 기능입니다.
finetune.py에서 '--add_kodata' 플래그를 주면 동작합니다.
"""
import random

from datasets import load_dataset, Dataset, concatenate_datasets


# 'MATH/PRM-800K' 12298  쓰지말자
# 'ARB' 713  쓰지말자
# 'scienceqa' 1317  여기서 100개 샘플링
# 'scibench' 616  여기서 100개 샘플링
# 'theoremqa' 564  쓰지말자
# 'reclor' 4530  여기서 100개 샘플링. 단, input이 앞 또는 뒤에 붙을 수 있음.
# 'tigerbot-kaggle' 386  쓰지 말자
# 'leetcode_ne' 1100  쓰지 말자
# 'airoboros' 2605  쓰지 말자
# 'guanaco' 797 쓰지 말자 (너무 김) -> 짧은 단어만 삭 모을까? -> no. ChatGPT 스타일이라 내가 원하는 스타일이 아님.
def load_platypus_ko():
    d = load_dataset("kyujinpy/KOpen-platypus")['train']
    d_classified = {}
    for e in d:
        if e['data_source'] not in d_classified:
            d_classified[e['data_source']] = []
        d_classified[e['data_source']].append(e)

    # scienceqa
    scienceqa = random.sample(d_classified['scienceqa'], k=100)
    for e in scienceqa:
        del(e['data_source'])
        e['input'] = ''

    # scibench
    scibench = random.sample(d_classified['scibench'], k=100)
    for e in scibench:
        del(e['data_source'])
        e['input'] == ''

    # reclor
    _reclor = random.sample(d_classified['reclor'], k=100)
    reclor = []
    for e in _reclor:
        del(e['data_source'])
        # input이 앞에 들어가거나, 뒤에 들어가거나, 들어가지 않도록.
        if random.random() < 0.33:
            reclor.append({
                'instruction': f"{e['input']} {e['instruction']}",
                'input': '',
                'output': e['output']
            })
        elif random.random() < 0.67:
            reclor.append({
                'instruction': f"{e['instruction']} {e['input']}",
                'input': '',
                'output': e['output']
            })
        else:
            reclor.append({
                'instruction': f"{e['instruction']}",
                'input': '',
                'output': e['output']
            })

    dataset_list = []
    for l in [scienceqa, scibench, reclor]:
        dataset_list.append(Dataset.from_list(l))

    return concatenate_datasets(dataset_list)


# OpenOrca-KO
# 'cot' 2117  전부 쓰자. 근데 input은 안 써도 될 듯.
# 'flan' 9434  쓰지말자 (만약 정 해볼 게 없을 경우 쓰자)
# 'KoCoT' 2159 여기서 100개 샘플링 (품질이 좋지는 않으나, 채점 시 번역투 데이터셋을 썼다면 이 데이터셋이 좋게 작용할 수 있어 보임)
# 'niv' 1571 쓰지 말자 (만약 정 해볼 게 없을 경우 쓰자)
# 't0' 6351  instruction과 output이 모두 200자 이하인 데이터만 골라 쓰자. (고르면 838개 남음)
def load_openorca_ko():
    orca = load_dataset("kyujinpy/OpenOrca-KO")['train']
    orca_by_name = {}
    for d in orca:
        if d['id'].split('.')[0] not in orca_by_name:
            orca_by_name[d['id'].split('.')[0]] = []
        orca_by_name[d['id'].split('.')[0]].append(d)

    # cot
    cot = []
    for d in orca_by_name['cot']:
        cot.append({
            'instruction': d['instruction'],
            'input': '',
            'output': d['output']
        })

    # KoCoT
    sampled_kocot = random.sample(orca_by_name['KoCoT'], k=100)
    KoCoT = []
    for d in sampled_kocot:
        KoCoT.append({
            'instruction': d['instruction'],
            'input': '',
            'output': d['output']
        })

    # 't0'
    # 30% 확률로 input이 앞에 포함되도록.
    reduced_t0 = [e for e in orca_by_name['t0'] if len(e['output']) < 150 and len(e['instruction']) < 150]
    t0 = []
    for d in reduced_t0:
        if random.random() > 0.7:
            KoCoT.append({
                'instruction': d['instruction'],
                'input': '',
                'output': d['output']
            })
        else:
            KoCoT.append({
                'instruction': f"{d['input']} {d['instruction']}",
                'input': '',
                'output': d['output']
            })

    dataset_list = []
    for l in [cot, KoCoT, t0]:
        dataset_list.append(Dataset.from_list(l))

    return concatenate_datasets(dataset_list)

















