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


# from verl.utils.hdfs_io import copy, makedirs
# from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


# def extract_solution(solution_str):
#     return remove_boxed(last_boxed_only_string(solution_str))


# def find_duplicates_between_datasets(dataset1, dataset2, key_field='problem'):
#     """
#     检测两个数据集之间基于指定字段的重复项
#     """
#     # 获取两个数据集的指定字段内容
#     set1 = set(dataset1[key_field])
#     set2 = set(dataset2[key_field])
    
#     # 找到重复项
#     duplicates = set1.intersection(set2)
    
#     return duplicates, len(duplicates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/math_train")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source_train = "data/DAPO-Math-17k-Processed/all/train-00000-of-00001.parquet"
    dataset_dapo = datasets.load_dataset("parquet", data_files=data_source_train, split="train")

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            _ = example.pop("source_prompt")
            question = example.pop("prompt").strip()
            extra_info = example.pop("extra_info")
            raw_index = extra_info["index"]
            data_source = example.pop("data_source")
            ability = example.pop("ability") if "ability" in example else None
            reward_model = example.pop("reward_model")
            answer = reward_model["ground_truth"]
            raw_reward_model_style = reward_model["style"]
            solution = example.pop("solution")

            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant."}, 
                    {"role": "user", "content": f"{question}\n" + instruction_following}
                ],
                "ability": ability,
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split, 
                    "index": idx, 
                    "raw_index": raw_index, 
                    "raw_rm_style": raw_reward_model_style,
                    "subject": ability, 
                    "solution": solution
                },
            }
            return data

        return process_fn

    train_dataset = dataset_dapo.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train_dapo.parquet"))

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)
