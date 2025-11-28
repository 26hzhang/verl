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


    dataset_gpqa = datasets.load_dataset("data/raw/gpqa", trust_remote_code=True)['train']

    # import pdb; pdb.set_trace()
    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            if data_source == "gpqa":
                question = example.pop('question')
                subject = example.pop('Subdomain')
                answer = example.pop('answer')
                solution = example.pop('Explanation')
            else:
                raise NotImplementedError
            data = {
                "data_source": data_source,
                "prompt": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"{question}\n" + instruction_following}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx, "subject": subject, "solution": solution},
            }
            return data

        return process_fn

    
    test_dataset_gpqa = dataset_gpqa.map(function=make_map_fn("test", "gpqa"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset_gpqa.to_parquet(os.path.join(local_dir, "test_gpqa.parquet"))


    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
