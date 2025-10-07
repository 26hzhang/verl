"""
Preprocess the Video-Holmes dataset to parquet format
"""

import argparse
import os
import sys
import datasets

home_dir = "/mnt/workspace/workgroup_dev/zhanghao/verl"
sys.path.append(home_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/mnt/workspace/workgroup_dev/zhanghao/verl/data/geo3k_multiturn_w_tool")
    args = parser.parse_args()

    data_path = "/mnt/workspace/workgroup_dev/zhanghao/verl/data/raw/Video-Holmes"
    train_dataset = datasets.load_dataset("json", data_files=os.path.join(data_path, "train_Video-Holmes.json"))["train"]
    test_dataset = datasets.load_dataset("json", data_files=os.path.join(data_path, "test_Video-Holmes.json"))["train"]

    data_source = "TencentARC/Video-Holmes"

    video_abs_path = "/mnt/workspace/workgroup_dev/zhanghao/verl/data/raw/Video-Holmes/videos_cropped"

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    # TODO: define it
    system_prompt = (
        r"You are a math expert. You are given a question and you need to solve it step by step. "
        r"Reasoning step by step before any tool call. "
        r"You should use the `calc_geo3k_reward` tool after step by step solving the question, "
        r"before generate final answer at least once and refine your answer if necessary. "
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("Question")  # str
            options = example.pop("Options")  # dict, {"A": "xxx", "B": "xxx", ...}
            answer = example.pop("Answer")  # option symbol, i.e., A or B or C ...
            explanation = example.pop("Explanation")
            video = example.pop("video ID")
            question_type = example.pop("Question Type")
            org_question_id = example.pop("Question ID")

            # process prompt by combine instructions, question, and options
            options = "\n".join([f"{key}. {value}" for key, value in options.items()])
            problem = f"{question}\nOptions:\n{options}"
            prompt = instruction_following + "\n" + problem

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "ability": question_type,
                "images": [],
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "org_index": org_question_id,
                    "answer": answer,
                    "question": question,
                    "options": options,
                    "explanation": explanation,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "calc_video_reward": {
                            "create_kwargs": {"ground_truth": answer},
                        },
                        "xxx_tool": {
                            "create_kwargs": {"video_path": video}
                        }
                    },
                },
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))
