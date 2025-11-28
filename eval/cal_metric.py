import argparse
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import csv
import os

def unbiased_pass_at_k_accuracy(file_path, k, n):
    # Read the JSONLines file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Group predictions by question_id
    grouped_data = defaultdict(list)
    for example in data:
        question_id = example["question_id"]
        if len(grouped_data[question_id]) >= n:
            continue
        grouped_data[question_id].append(example)

    # Calculate unbiased pass@K accuracy
    total_pass_k_prob = 0
    total_questions = 0
    total_acc = 0
    for question_id, examples in grouped_data.items():
        # Count correct and total solutions for this question
        assert len(examples) == n  # Total number of samples
        correct_labels = [1 if ex["label"] else 0 for ex in examples ]
        c = sum(correct_labels)  # Number of correct samples
        # Apply unbiased pass@k formula
        assert n >= k
        # Calculate unbiased pass@k probability
        if n - c < k:
            prob = 1.0
        else:
            prob = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
            
        total_pass_k_prob += prob
        total_acc += sum(correct_labels[:k])
        total_questions += 1
    
    # Calculate average pass@K probability across all questions
    avg_pass_k = total_pass_k_prob / total_questions if total_questions > 0 else 0
    avg_acc = total_acc / total_questions / k if total_questions > 0 else 0
    # print(f"Questions Number: {total_questions}, Unbiased Pass@{k}/{n}: {avg_pass_k}, Accuracy@{k}/{n}: {avg_acc}")
    return avg_pass_k, avg_acc, total_questions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to the JSONLines file")
    args = parser.parse_args()
   
    Ks = [1, 2, 4, 8, 16,] #  32, 64, 128, 256
    print(f"Calculating Unbiased Pass@K for {args.file_path}")
    all_result_pass = []
    all_result_acc = []
    all_result_questions = []
    for K in tqdm(Ks):
        # print("-" * 80)
        test_file = args.file_path
        unbiased_pass_k_accuracy, avg_acc, total_questions = unbiased_pass_at_k_accuracy(test_file, k=K, n=16)
        all_result_pass.append(unbiased_pass_k_accuracy)
        all_result_acc.append(avg_acc)
        all_result_questions.append(total_questions)
    print("Pass@k\n", all_result_pass)
    print("Questions Number\n", all_result_questions)

    print("Acc@k\n", all_result_acc)
        # 写入CSV文件
    data_name = args.file_path.split("/")[-1].split("-test")[0]
    filename_pass = f'{data_name}_pass_k.csv'
    filename_acc = f'{data_name}_acc_k.csv'
    if not os.path.exists(filename_pass):
    # 如果文件不存在，创建文件并写入表头
        with open(filename_pass, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model", 1, 2, 4, 8, 16])
    with open(filename_pass, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.file_path] + all_result_pass)
    
    
    if not os.path.exists(filename_acc):
    # 如果文件不存在，创建文件并写入表头
        with open(filename_acc, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["model", 1, 2, 4, 8, 16])
    with open(filename_acc, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.file_path] + all_result_acc)