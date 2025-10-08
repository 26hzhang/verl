"""
Preprocess the Video-Holmes dataset to parquet format
"""

import argparse
import os
import sys
from torchcodec.decoders import VideoDecoder

import datasets
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    args = parser.parse_args()

    json_path = args.json_path
    dataset = datasets.load_dataset("json", data_files=json_path)["train"]

    data_source = "TencentARC/Video-Holmes"

    video_abs_path = args.video_dir

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    # TODO: define it


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):

            system_prompt = (
                r"You are a video QA expert. You are given a question and you need to solve it step by step. "
                r"You should actively use `extract_frames`, `zoom_in_frame` and `verify_answer` according to their usage scenarios and please reasoning step by step before any tool call. "
                r"At least one tool call should be made before generate final answer, and refine your answer if necessary. "
            )

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

            video_path = os.path.join(video_abs_path, f"{video}.mp4")
            if not os.path.exists(video_path):
                print(f"Video {video_path} not found")
                return None

            vr = VideoDecoder(video_path)
            duration = vr.metadata.duration_seconds
            # duration to HH:MM:SS
            duration = time.strftime("%H:%M:%S", time.gmtime(duration))

            prompt = instruction_following + "\n" + problem + "\n" + f"Let's start with `extract_frames()` from start to end to understand the big picture first, the duration of the video is {duration}."

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
                        "verify_answer": {
                            "create_kwargs": {"ground_truth": answer},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                        "extract_frames": {
                            "create_kwargs": {
                                "video_path": os.path.join(video_abs_path, f"{video}.mp4"),
                                "num_frames": 4,
                                "size": 256,
                            },
                        },
                        "zoom_in_frame": {
                            "create_kwargs": {
                                "size": 512,
                            },
                        },
                    },
                }
            }
            return data
        return process_fn

    dataset = dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    dataset.to_parquet(args.output_path)
