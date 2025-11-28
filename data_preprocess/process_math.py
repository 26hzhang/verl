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
    parser.add_argument("--local_dir", default="data/math")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source_train = "data/math/hendrycks_math"
    data_source_test = "data/math/MATH-500"
    data_source = "hendrycks_math:train_MATH-500:test"
    # print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset_train1 = datasets.load_dataset(data_source_train, 'algebra', trust_remote_code=True)
    dataset_train2 = datasets.load_dataset(data_source_train, 'counting_and_probability', trust_remote_code=True)
    dataset_train3 = datasets.load_dataset(data_source_train, 'geometry', trust_remote_code=True)
    dataset_train4 = datasets.load_dataset(data_source_train, 'intermediate_algebra', trust_remote_code=True)
    dataset_train5 = datasets.load_dataset(data_source_train, 'number_theory', trust_remote_code=True)
    dataset_train6 = datasets.load_dataset(data_source_train, 'prealgebra', trust_remote_code=True)
    dataset_train7 = datasets.load_dataset(data_source_train, 'precalculus', trust_remote_code=True)
    # 提取训练集（假设都有'train'分割）
    hendrycks_math_train = [
        dataset_train1['train'],
        dataset_train2['train'], 
        dataset_train3['train'],
        dataset_train4['train'],
        dataset_train5['train'],
        dataset_train6['train'],
        dataset_train7['train']
    ]
    hendrycks_math_test = [
        dataset_train1['test'],
        dataset_train2['test'], 
        dataset_train3['test'],
        dataset_train4['test'],
        dataset_train5['test'],
        dataset_train6['test'],
        dataset_train7['test']
    ]

    # 合并所有训练集
    merged_hendrycks_math_train = concatenate_datasets(hendrycks_math_train)
    merged_hendrycks_math_test = concatenate_datasets(hendrycks_math_test)

    dataset_test = datasets.load_dataset(data_source_test, trust_remote_code=True)['test']
    duplicates_train, len_duplicates_train = find_duplicates_between_datasets(merged_hendrycks_math_train, dataset_test, key_field='problem')
    duplicates_test, len_duplicates_test = find_duplicates_between_datasets(merged_hendrycks_math_test, dataset_test, key_field='problem')
    print(f"重复训练集：{len_duplicates_train}")
    print(f"重复测试集：{len_duplicates_test}")
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question
            subject = example.pop("subject") if 'subject' in example else example.pop("type")
            solution = example.pop("solution")
            answer = extract_solution(solution)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "system", "content": instruction_following}, {"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx, "subject": subject, "solution": solution},
            }
            return data

        return process_fn

    train_dataset = merged_hendrycks_math_train.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = dataset_test.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
