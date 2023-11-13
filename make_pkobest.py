import sys
from random import choice
from functools import partial
from pprint import pprint

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict

from korean_utils import bojosa
from templates import prompts


def merge_train_valid(ds):
    return concatenate_datasets([ds['train'], ds['validation']])

def load_custom_dataset(dataset_name):
    # for kobest dataset, temporarily use the both train and validation split in this time.
    if dataset_name == 'boolq':
        ds = load_dataset('skt/kobest_v1', 'boolq')
        return merge_train_valid(ds)

    elif dataset_name == 'copa':
        ds = load_dataset('skt/kobest_v1', 'copa')
        return merge_train_valid(ds)

    elif dataset_name == 'hellaswag':
        ds = load_dataset('skt/kobest_v1', 'hellaswag')
        return merge_train_valid(ds)

    elif dataset_name == 'sentineg':
        ds = load_dataset('skt/kobest_v1', 'sentineg')
        return merge_train_valid(ds)

    elif dataset_name == 'wic':
        ds = load_dataset('skt/kobest_v1', 'wic')
        return merge_train_valid(ds)

    else:
        raise NotImplementedError


def make_my_dataset(debug, mode, truncate_max=None):
    assert 'my' in mode or 'baseline' in mode

    # you can modify below
    target_dataset_names = ['hellaswag', 'copa', 'boolq', 'sentineg', 'wic']

    # making prompts for each dataset with randomly chosen template
    prompts_collection = {k: [] for k in target_dataset_names}
    for dataset_name in target_dataset_names:
        ds = load_custom_dataset(dataset_name)
        custom_templates = prompts.datasets[dataset_name]
        assert len(custom_templates) == 10
        for i, row in enumerate(ds):
            template = choice(custom_templates)
            prompt = getattr(prompts, f"_process_{dataset_name}")(template, **row)
            prompts_collection[dataset_name].append(prompt)
            if truncate_max and i >= truncate_max:
                break  # slicing needs too many code

    def list_to_dataset(l):
        d = {k: [] for k in l[0].keys()}
        for e in l:
            for k, v in e.items():
                d[k].append(v)
        return Dataset.from_dict(d)

    prompts_datasets = {k: list_to_dataset(v) for k, v in prompts_collection.items()}

    if debug:
        print("printing each dataset...")
        for k, v in prompts_datasets.items():
            print("dataset name: ", k)
            pprint(v[0])

    kullm_v2 = load_dataset('nlpai-lab/kullm-v2')['train']
    if 'my' in mode:
        # collection = concatenate_datasets([
        #     kullm_v2, prompts_boolq, prompts_copa, prompts_hellaswag, prompts_sentineg, prompts_wic])
        collection = concatenate_datasets([kullm_v2] + [d for d in prompts_datasets.values()])
    elif 'baseline' in mode:
        collection = concatenate_datasets([kullm_v2])
    else:
        raise NotImplementedError
    collection = collection.shuffle(seed=42)

    if debug:
        collection = collection[:1000]
        collection = Dataset.from_dict(collection)

    return DatasetDict({
        "train": collection
    })


def main():
    # collection = make_my_dataset()
    # print(collection)
    raise AssertionError("main must not be called")


if __name__ == '__main__':
    main()
