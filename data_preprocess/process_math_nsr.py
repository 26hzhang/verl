# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import os
from datasets import concatenate_datasets
import datasets


from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def find_duplicates_between_datasets(dataset1, dataset2, key_field='problem'):
    """
    检测两个数据集之间基于指定字段的重复项
    """
    # 获取两个数据集的指定字段内容
    set1 = set(dataset1[key_field])
    set2 = set(dataset2[key_field])
    
    # 找到重复项
    duplicates = set1.intersection(set2)
    
    return duplicates, len(duplicates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/math_train")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source_train = "/mindopt/weiwen/RLVR-Decomposed-main/data/math"
    data_source_amc23 = "/mindopt/weiwen/RLVR-Decomposed-main/data/amc23"
    data_source_aime2025 = "/mindopt/weiwen/RLVR-Decomposed-main/data/aime2025"
    data_source_aime2024 = "/mindopt/nlp_nas_2/weiwen/cut2_dynamic/data/math_test/AIME2024"
    # data_source = "hendrycks_math"
    # print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset_math = datasets.load_dataset(data_source_train,trust_remote_code=True)
    dataset_amc23_test = datasets.load_dataset(data_source_amc23,trust_remote_code=True)['test']
    dataset_aime2025_test = datasets.load_dataset(data_source_aime2025,trust_remote_code=True)['test']
    dataset_aime2024_test = datasets.load_dataset(data_source_aime2024,trust_remote_code=True)['test']
    dataset_math_train = dataset_math['train']
    dataset_math_test = dataset_math['test']

   


    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            if data_source == "aime2024":
                question = example.pop("problem")
                answer = example.pop("answer")
                solution = example.pop("solution")
                subject = 'math'
            else:
                reward_model = example.pop("reward_model")
                ground_truth_tuple = reward_model.pop("ground_truth")
                extra_info = example.pop("extra_info")
                question = ground_truth_tuple['question']
                subject = example.pop("type") if "type" in example else None
                answer = ground_truth_tuple['target']
                if split == "test":
                    solution = None
                else:
                    solution = ground_truth_tuple.pop("solution")

            data = {
                "data_source": data_source,
                "prompt": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"{question}\n" + instruction_following}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx, "subject": subject, "solution": solution},
            }
            return data

        return process_fn

    train_dataset = dataset_math_train.map(function=make_map_fn("train", "hendrycks_math"), with_indices=True)
    test_dataset_math = dataset_math_test.map(function=make_map_fn("test", "hendrycks_math"), with_indices=True)
    test_dataset_amc23 = dataset_amc23_test.map(function=make_map_fn("test", "amc23"), with_indices=True)
    test_dataset_aime2025 = dataset_aime2025_test.map(function=make_map_fn("test", "aime2025"), with_indices=True)
    test_dataset_aime2024 = dataset_aime2024_test.map(function=make_map_fn("test", "aime2024"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train_math.parquet"))
    test_dataset_math.to_parquet(os.path.join(local_dir, "test_math.parquet"))
    test_dataset_amc23.to_parquet(os.path.join(local_dir, "test_amc23.parquet"))
    test_dataset_aime2025.to_parquet(os.path.join(local_dir, "test_aime2025.parquet"))
    test_dataset_aime2024.to_parquet(os.path.join(local_dir, "test_aime2024.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
