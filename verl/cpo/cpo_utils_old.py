from collections import defaultdict
from tensordict import TensorDict
import numpy as np
import torch
from verl import DataProto


def compute_difficulty_mask(batch):
    """
    Identify the all correct and all incorrect samples in the batch based on each sampele's rollouts
    """
    reward_tensor_per_example = batch.batch["token_level_scores"].sum(dim=-1)
    # ref.: verl/utils/reward_score/math_cpo.py to check the reward calculation logic
    min_reward = reward_tensor_per_example.min().item()  # -1.0
    max_reward = reward_tensor_per_example.max().item()  #  1.0
    uid2mean_reward = {}
    for i in range(batch.non_tensor_batch["uid"].shape[0]):
        uid = batch.non_tensor_batch["uid"][i]
        reward_one = reward_tensor_per_example[i].item()
        if uid not in uid2mean_reward:
            uid2mean_reward[uid] = []
        uid2mean_reward[uid].append(reward_one)
    difficulty_mask = torch.zeros_like(reward_tensor_per_example)
    # TODO: 如果reward_tensor_per_example所有样本都全对或者全错，那这里的逻辑可能会有问题
    for i in range(batch.non_tensor_batch["uid"].shape[0]):
        uid = batch.non_tensor_batch["uid"][i]
        mean_reward = uid2mean_reward[uid]
        if mean_reward == min_reward:
            difficulty_mask[i] = -1
        elif mean_reward == max_reward:
            difficulty_mask[i] = 1
    return difficulty_mask


def sample_by_data_source(
    data_source, sample_inputs, sample_outputs, golds, sample_scores, 
    samples_per_source: int = 3, random_seed: int = 42) -> tuple:
    """
    从每个数据源中采样指定数量的样本
    
    Args:
        data_source: 数据源标识列表
        sample_inputs: 输入样本列表
        sample_outputs: 输出样本列表
        golds: 金标准答案列表
        sample_scores: 分数列表
        samples_per_source: 每个数据源采样的数量，默认3条
        random_seed: 随机种子，确保可重复性
        
    Returns:
        tuple: (采样后的inputs, outputs, golds, scores, data_sources)
    """
    import random
    
    # 设置随机种子确保可重复性
    random.seed(random_seed)
    
    # 按数据源分组索引
    source_indices = defaultdict(list)
    for idx, source in enumerate(data_source):
        source_indices[source].append(idx)
    
    # 存储采样结果
    sampled_indices = []
    
    print(f"find {len(source_indices)} different data source in eval:")
    
    # 从每个数据源中采样
    for source, indices in source_indices.items():
        available_count = len(indices)
        sample_count = min(samples_per_source, available_count)
        
        # 随机采样
        sampled_idx = random.sample(indices, sample_count)
        sampled_indices.extend(sampled_idx)
        
        print(f"- {source}: {available_count} in total, sample {sample_count}")
    
    # 根据采样索引提取对应数据
    sampled_inputs = [sample_inputs[i] for i in sampled_indices]
    sampled_outputs = [sample_outputs[i] for i in sampled_indices]
    sampled_golds = [golds[i] for i in sampled_indices]
    sampled_scores = [sample_scores[i] for i in sampled_indices]
    
    print(f"\Totally, {len(sampled_indices)} samples are logged")
    
    return sampled_inputs, sampled_outputs, sampled_golds, sampled_scores


def find_subsequence_position(tensor, subsequence):
    """
    Find the position of a subsequence in a tensor.
    
    Args:
        tensor: Input tensor to search in
        subsequence: List or tensor of the subsequence to find
        
    Returns:
        Position index if found, -1 if not found
    """
    if isinstance(subsequence, list):
        subsequence = torch.tensor(subsequence, dtype=tensor.dtype)
    
    subseq_len = len(subsequence)
    tensor_len = len(tensor)
    
    if subseq_len > tensor_len:
        return -1
    
    # 滑动窗口搜索
    for i in range(tensor_len - subseq_len + 1):
        if torch.equal(tensor[i:i + subseq_len], subsequence):
            return i
    
    return -1


def compute_judge(data, tokenizer, gold_as_hint=True):
    """
    Compute judgement for a batch of data.
    Args:
        data: DataProto object containing the input data.
        mask: Mask for the input data (bad cases).
        tokenizer: tokenizer
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    if gold_as_hint:
        judgments = []
        for i in range(len(data)):
            data_item = data[i]
            solution = data_item.non_tensor_batch['extra_info']["solution"]
            judgments.append(solution)
        return judgments
    else:
        raise NotImplementedError
        # all_judge_inputs = []
        # valid_indices = []  # 跟踪哪些原始索引对应mask==1
        # judgment_outputs = {}
        # # prepare judge data
        # for i in range(len(data)):
        #     data_item = data[i]  # DataProtoItem
        #     valid_indices.append(i)  # 记录原始索引
        #     prompt_ids = data_item.batch["prompts"]
        #     prompt_length = prompt_ids.shape[-1]
        #     solution = data_item.non_tensor_batch['extra_info']["solution"]
        #     valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        #     valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        #     response_ids = data_item.batch["responses"]
        #     valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        #     valid_response_ids = response_ids[:valid_response_length]

        #     # decode
        #     prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        #     prompt_str = prompt_str.split("\nuser\n")[-1].split("\nassistant\n")[0]
        #     response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        #     judge_text = f"###Question: {prompt_str}\n###Model response: {response_str}\n###Ground truth: {solution}\nPlease analyze the above model response to the given question and identify any errors or issues. I have also provided the ground-truth solution for reference, but do not mention or disclose that you have access to it in your analysis. Please show your analysis concisely inside <judge></judge> tags in your reply."
        #     all_judge_inputs.append(judge_text)
        # llm_client = gpt_judge.openai_llm()
        # message_list = [[{"role": "user","content": x}]   for x in all_judge_inputs] 
        
        # ### generate judgment 
        # judge_outputs = llm_client.generate_outputs(message_list)

        # # post processing judgment 
        # failed_count = 0
        # # 遍历judge_outputs，其索引对应valid_indices
        # for judge_idx, gpt_output in enumerate(judge_outputs):
        #     original_data_idx = valid_indices[judge_idx]  # 对应原始data中的索引
        #     # 检查judgment是否成功
        #     if re.search(r"<judge>.*?</judge>", gpt_output, re.DOTALL):
        #         judge_content = re.search(r"<judge>(.*?)</judge>", gpt_output, re.DOTALL).group(1)
        #         judge_content = judge_content.strip()
        #         # 检查是否是失败的judgment
        #         if judge_content == "FAIL" or not judge_content:
        #             judgment_outputs[original_data_idx] = "NA"
        #             failed_count += 1
        #         else:
        #             judgment_outputs[original_data_idx] = judge_content
        #     else:
        #         judgment_outputs[original_data_idx] = "NA"
        #         failed_count += 1
        # judgments = [ judgment_outputs[i] for i in range(len(data)) ]
        # return judgments



def add_hint_to_data_batch(data, hints, tokenizer, hint_type='gold'):
    """
    Add hints to a batch of DataProtoItems.
    
    Args:
        data: DataProto object containing multiple DataProtoItems
        hints: List of hint strings, same length as data, or single string for all items
        tokenizer: Tokenizer to encode hints
    
    Returns:
        New DataProto object with modified items
    """
    if isinstance(hints, str):
        # 如果是单个hint，为所有item使用相同的hint
        hints = [hints] * len(data)
    
    if len(hints) != len(data):
        raise ValueError(f"Number of hints ({len(hints)}) must match number of data items ({len(data)})")
    
    # 创建新的数据列表
    non_tensor_batch = data.non_tensor_batch

    new_data = {
        "prompts": [],
        "responses": [],
        "input_ids": [],  # here input_ids become the whole sentences
        "attention_mask": [],
        "position_ids": [],
    }
    unique_new_data = {
        "prompts": [],
        "responses": [],
        "input_ids": [],  # here input_ids become the whole sentences
        "attention_mask": [],
        "position_ids": [],
    }
    # breakpoint()
    for i, (data_item, hint_text) in enumerate(zip(data, hints)):
        original_input_ids = data_item.batch["input_ids"]
        original_attention_mask = data_item.batch["attention_mask"]
        original_position_ids = data_item.batch["position_ids"]
        responses = data_item.batch["responses"]
        response_mask = data_item.batch["response_mask"]
        prompts = data_item.batch["prompts"]
        prompts_text = tokenizer.decode(prompts)
        responses_text = tokenizer.decode(responses)
        prompt_length = prompts.shape[-1]
        original_input_ids_length = original_input_ids.shape[-1]

        # breakpoint()
        if hint_type == 'gold':
            if "llama" in tokenizer.name_or_path.lower():
                user_tag_toks = [128006, 882, 128007, 271]
                # insert_index = find_subsequence_position(prompts, user_tag_toks) 
                insert_index = prompt_length
                max_hint_length = (prompts != 128009).nonzero(as_tuple=True)[0][0].item()
                assert insert_index >= 0, "Special tag not found in prompt"
                hint_prompt = prompts_text.split("You are a helpful assistant.<|eot_id|>")[1].split("Please reason step by step, and put your final answer within \\boxed{}")[0]
                hint_prompt += 'Please refer to the above solution to derive your answer, reason step by step, and put your final answer within \\boxed{}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
                hint_prompt_tok = tokenizer.encode(hint_prompt, add_special_tokens=False)
                hint_text = f"{hint_text}<|eot_id|>" 
                hint_text_tok = tokenizer.encode(hint_text, add_special_tokens=False)
                
            elif "qwen" in tokenizer.name_or_path.lower():
                user_tag_toks = [151644, 872, 198]
                assistant_tag_toks = [ 151644, 77091, 198]
                insert_index = find_subsequence_position(prompts, assistant_tag_toks)
                max_hint_length = (prompts != 151643).nonzero(as_tuple=True)[0][0].item()
                assert insert_index >= 0, "Special tag not found in prompt"
                hint_prompt = prompts_text.split("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")[1].split("Please reason step by step, and put your final answer within \\boxed{}")[0]
                hint_prompt += 'Please refer to the above solution to derive your answer, reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n'
                hint_prompt_tok = tokenizer.encode(hint_prompt)
                hint_text = f"<|im_start|>assistant\n{hint_text}<|im_end|>\n" 
                hint_text_tok = tokenizer.encode(hint_text)
                # breakpoint()

        else:
            raise NotImplementedError
            # hint_prompt = "\nYou can use the following hint to derive your answer, but never mentioned that you are using the hint:\n"
            # hint_prompt_tok = tokenizer.encode(hint_prompt)
        # breakpoint()

        
        # 编码hint文本
        hint_tok = hint_text_tok[-(max_hint_length - len(hint_prompt_tok)):] + hint_prompt_tok
        hint_ids = torch.tensor(hint_tok, dtype=torch.int64)
        hint_length = hint_ids.shape[-1]
        new_input_ids = torch.cat([original_input_ids[:insert_index], hint_ids, original_input_ids[insert_index:]])
        new_input_ids = new_input_ids[-original_input_ids_length:]
        assert (new_input_ids[prompt_length:] == original_input_ids[prompt_length:]).all()

        new_attention_mask = torch.cat([original_attention_mask[:insert_index], torch.ones(hint_length, dtype=torch.int64), original_attention_mask[insert_index:]])
        new_attention_mask = new_attention_mask[-original_input_ids_length:]

        # 处理 position_ids
        # 获取插入位置的 position 值
        insert_position = original_position_ids[insert_index]
        # 创建 hint 部分的 position_ids，从插入位置开始连续递增
        hint_position_ids = torch.arange(insert_position, insert_position + hint_length, dtype=torch.int64)
        # 调整插入位置之后的 position_ids，需要加上 hint_length 的偏移
        adjusted_later_positions = original_position_ids[insert_index:] + hint_length

        new_position_ids = torch.cat([original_position_ids[:insert_index], hint_position_ids, adjusted_later_positions])
        new_position_ids = new_position_ids[-original_input_ids_length:]

        new_prompts = torch.cat([prompts[:insert_index], hint_ids, prompts[insert_index:]])
        new_prompts = new_prompts[-prompt_length:]
        
        new_data["prompts"].append(new_prompts)
        new_data["responses"].append(responses)
        new_data["input_ids"].append(new_input_ids)
        new_data["attention_mask"].append(new_attention_mask)
        new_data["position_ids"].append(new_position_ids)
    new_batch =  {
            "prompts": torch.stack(new_data["prompts"], dim=0),
            "responses": torch.stack(new_data["responses"], dim=0),
            "input_ids": torch.stack(new_data["input_ids"], dim=0),
            "attention_mask": torch.stack(new_data["attention_mask"], dim=0),
            "position_ids": torch.stack(new_data["position_ids"], dim=0),
        }
    _, unique_indices = np.unique(non_tensor_batch['uid'], return_index=True)    
    new_batch_gen = {
        "input_ids":new_batch["input_ids"][unique_indices, :prompt_length],
        "attention_mask":new_batch["attention_mask"][unique_indices, :prompt_length],
        "position_ids":new_batch["position_ids"][unique_indices, :prompt_length],
    }
    new_batch = TensorDict(
        new_batch,
        batch_size=new_batch["input_ids"].size(0),
        )
    new_batch_gen = TensorDict(
        new_batch_gen,
        batch_size=new_batch_gen["input_ids"].size(0),
        )
    new_data = DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch)
    new_data_gen = DataProto(batch=new_batch_gen, non_tensor_batch={x: non_tensor_batch[x][unique_indices] for x in non_tensor_batch})
    # 返回新的DataProto对象（假设DataProto有类似list的接口）
    return new_data, new_data_gen


def convert_tokens_with_logprobs(batch, tokenizer, mask_padding=True, config=None):
    """
    将token IDs还原为原始tokens，并与对应的log probabilities组合
    
    Args:
        batch: TensorDict包含responses, gw_yl_log_probs, old_log_probs, response_mask等
        tokenizer: 用于解码token IDs的tokenizer
        mask_padding: 是否使用response_mask来过滤padding tokens
    
    Returns:
        List of lists: 每个样本包含(token, gw_yl_log_probs, old_log_prob)的元组列表
    """
    responses = batch['responses']  # [256, 4096]
    gw_yl_log_probs = batch['gw_yl_log_probs']  # [256, 4096]
    old_log_probs = batch['old_log_probs']  # [256, 4096]
    response_mask = batch.get('response_mask', None)  # [256, 4096]

    
    batch_size, seq_len = responses.shape
    result = []
    
    for batch_idx in range(batch_size):
        sample_result = []
        for token_idx in range(seq_len):
            # 如果有response_mask且该位置为0（padding），则跳过
            if mask_padding and response_mask is not None and response_mask[batch_idx, token_idx] == 0:
                continue
                
            token_id = responses[batch_idx, token_idx].item()
            gw_yl_log_prob = gw_yl_log_probs[batch_idx, token_idx]
            old_logprob = old_log_probs[batch_idx, token_idx]
            ratio = torch.exp(gw_yl_log_prob - old_logprob)
            # 将token ID转换为token文本
            try:
                token_text = tokenizer.decode([token_id])
            except Exception as e:
                token_text = f"<UNK:{token_id}>"  # 如果解码失败，显示原始ID
            
            sample_result.append((token_text, torch.exp(gw_yl_log_prob), torch.exp(old_logprob),  ratio, (ratio > config.cpo_lambda) or ratio < 1/ config.cpo_lambda) )
        result.append(sample_result)
    
    return result


def process_token_data_to_string(token_data, info_format="<cpo>[|id:={index}|{token}|p(y|x,j):={val1:.4f}|p(y|x):={val2:.4f}|ratio:={val3:.4f}]</cpo>"):
    """
    将token数据处理成字符串，当最后一维为True时插入详细信息
    
    Args:
        token_data: 包含(token, tensor1, tensor2, tensor3, bool_tensor)元组的列表
        info_format: 信息插入的格式字符串，默认格式为[token|val1|val2|val3]
        
    Returns:
        str: 处理后的字符串
    """
    result_parts = []
    
    for i, (token, val1, val2, val3, should_insert) in enumerate(token_data):
        # 提取tensor的值
        val1_item = val1.item() if hasattr(val1, 'item') else val1
        val2_item = val2.item() if hasattr(val2, 'item') else val2
        val3_item = val3.item() if hasattr(val3, 'item') else val3
        # val4_item = val4.item() if hasattr(val4, 'item') else val4
        should_insert_item = should_insert.item() if hasattr(should_insert, 'item') else should_insert
        
        if should_insert_item:
            # 如果最后一维为True，插入详细信息
            info_str = info_format.format(
                token=token,
                val1=val1_item,
                val2=val2_item,
                val3=val3_item,
                # val4=val4_item,
                index=i
            )
            result_parts.append(info_str)
        else:
            # 否则只添加token文本
            result_parts.append(token)
    
    return ''.join(result_parts)
