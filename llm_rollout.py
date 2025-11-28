import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils import extract_answer_math
from grader import math_equal
from transformers import AutoTokenizer

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def prepare_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Qwen3-4B inference rollout for math problems")
    parser.add_argument("--model_path", type=str, default="model_weights/Qwen3-4B",
                        help="Path to model")
    parser.add_argument("--input_file", type=str, default="data/math_train/train_dapo.parquet",
                        help="Input parquet file")
    parser.add_argument("--output_file", type=str, default="math_dapo_qwen3-4b-rollout_n_10.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum tokens to generate")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=float, default=-1,
                        help="Top-k sampling parameter")
    parser.add_argument("--num_rollouts", type=int, default=10,
                        help="Number of rollouts per sample")
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume from this sample index")

    args = parser.parse_args()
    print(f"Arguments: {args}")

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Model {args.model_path} not found. Please check the path.")
        return

    # Load the model using vLLM
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        args.model_path,
        tensor_parallel_size=args.num_gpus,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        n=args.num_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load dataset using datasets library
    print(f"Loading dataset from: {args.input_file}")
    dataset = load_dataset('parquet', data_files=args.input_file, split='train')
    print(f"Dataset loaded: {len(dataset)} samples")

    # Resume support
    if args.resume_from > 0:
        print(f"Resuming from index {args.resume_from}")
        dataset = dataset.select(range(args.resume_from, len(dataset)))

    # Check if output file exists
    write_mode = 'w' if args.resume_from == 0 else 'a'
    if write_mode == 'w' and os.path.exists(args.output_file):
        user_input = input(f"Output file {args.output_file} already exists. Overwrite? (y/n): ")
        if user_input.lower() != 'y':
            print("Aborted.")
            return

    # Generate responses in batches
    print(f"Generating {args.num_rollouts} rollouts per sample...")

    # Track statistics
    total_correct = 0
    total_rollouts = 0
    sample_pass_rates = []

    with open(args.output_file, write_mode) as f:
        for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
            batch = dataset[i:i + args.batch_size]
            
            batch_prompts = [prepare_prompt(tokenizer, p) for p in batch['prompt']]

            # Generate using vLLM
            outputs = llm.generate(batch_prompts, sampling_params=sampling_params, use_tqdm=True)

            # Extract all rollouts
            results = [[output.outputs[j].text for j in range(len(output.outputs))] for output in outputs]
            assert len(batch_prompts) == len(results), f"Number of batch is not equal to number of results, got {len(batch_prompts)} != {len(results)}"
            assert len(results[0]) == args.num_rollouts, f"Number of generations is not equal to {args.num_rollouts}, got {len(results[0])}"

            # Grade and calculate pass rate
            for batch_idx in range(len(batch_prompts)):
                gold_answer = batch["reward_model"][batch_idx]["ground_truth"]

                # Grade all rollouts for this sample
                rollout_labels = []
                rollout_pred_answers = []

                for rollout_idx in range(args.num_rollouts):
                    response = results[batch_idx][rollout_idx]
                    pred_answer = extract_answer_math(response)
                    rollout_pred_answers.append(pred_answer)
                    # Compare with ground truth
                    label = math_equal(pred_answer, gold_answer, timeout=True)
                    rollout_labels.append(label)

                pass_rate = sum(rollout_labels) / args.num_rollouts if args.num_rollouts > 0 else 0.0

                sample_pass_rates.append(pass_rate)
                total_correct += sum(rollout_labels)
                total_rollouts += args.num_rollouts

                # Construct output record
                vllm_sample = {
                    "prompt": batch["prompt"][batch_idx],
                    "level": None if "dapo" in args.input_file else batch["level"][batch_idx],
                    "data_source": batch["data_source"][batch_idx],
                    "ability": batch["ability"][batch_idx],
                    "reward_model": batch["reward_model"][batch_idx],
                    "extra_info": batch["extra_info"][batch_idx],
                    "rollout": {
                        "n": args.num_rollouts,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "pass_rate": pass_rate,
                        "rollout_labels": rollout_labels,
                        "rollout_answers": rollout_pred_answers,
                        "rollout_responses": results[batch_idx],
                    }
                }
                if "dapo" in args.input_file:
                    vllm_sample.pop("level")  # DAPO do not have this key
                f.write(json.dumps(vllm_sample, ensure_ascii=False) + '\n')

            f.flush()

    # Print statistics
    print(f"\n✓ Rollout completed! Results saved to: {args.output_file}")
    print(f"\n=== Statistics ===")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Rollouts per sample: {args.num_rollouts}")
    print(f"  Total rollouts: {total_rollouts}")
    print(f"  Total correct: {total_correct}")
    print(f"  Overall accuracy: {total_correct / total_rollouts * 100:.2f}%")
    print(f"  Average pass rate: {sum(sample_pass_rates) / len(sample_pass_rates) * 100:.2f}%")
    print(f"  100% pass rate: {sum(1 for pr in sample_pass_rates if pr == 1.0)} samples ({sum(1 for pr in sample_pass_rates if pr == 1.0) / len(sample_pass_rates) * 100:.2f}%)")
    print(f"    0% pass rate: {sum(1 for pr in sample_pass_rates if pr == 0.0)} samples ({sum(1 for pr in sample_pass_rates if pr == 0.0) / len(sample_pass_rates) * 100:.2f}%)")


if __name__ == "__main__":
    main()
