import torch


def advantage_wrap_naive(data, advantages, entropys, config):
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    # all-wrong examples using cpo
    old_log_probs = data.batch["old_log_probs"]
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    ratio = torch.clamp(ratio, 1/ config.cpo_lambda, config.cpo_lambda)
    gw_yl_prob_clip_mask = torch.exp(gw_yl_log_probs) > 0.1
    old_prob_clip_mask = torch.exp(old_log_probs) > 0.1
    ratio_mask_pos = ratio >= config.cpo_lambda
    ratio_mask_neg = ratio <= 1 / config.cpo_lambda
    combined_mask_pos = ratio_mask_pos & data.batch["response_mask"].bool() & gw_yl_prob_clip_mask
    combined_mask_neg = ratio_mask_neg & data.batch["response_mask"].bool() & old_prob_clip_mask
    # for combined_mask_pos, should slightly increase the adv, the higher entropy, increase more adv
    if config.pos_alpha:
        advantages[combined_mask_pos] = advantages[combined_mask_pos] + config.pos_alpha * entropys[combined_mask_pos]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.neg_alpha:
        entropy_credit_neg = entropys.max(dim=1, keepdim=True).values - entropys
        advantages[combined_mask_neg] = advantages[combined_mask_neg] - config.neg_alpha * entropy_credit_neg[combined_mask_neg]
    return advantages


def advantage_wrap_naive_qwen3(data, advantages, entropys, config):
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    grpo_calculation_mask = data.batch["response_mask"]
    # all-wrong examples using cpo
    old_log_probs = data.batch["old_log_probs"]
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    ratio = torch.clamp(ratio, 1/ config.cpo_lambda, config.cpo_lambda)
    gw_yl_prob_clip_mask = torch.exp(gw_yl_log_probs) > 0.1
    old_prob_clip_mask = torch.exp(old_log_probs) > 0.1
    ratio_mask_pos = ratio >= config.cpo_lambda
    ratio_mask_neg = ratio <= 1 / config.cpo_lambda
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, ratio.size(-1))
    combined_mask_pos = ratio_mask_pos & data.batch["response_mask"].bool() & gw_yl_prob_clip_mask & (yl_rewards_expanded == 1)
    combined_mask_neg = ratio_mask_neg & data.batch["response_mask"].bool() & old_prob_clip_mask & (yl_rewards_expanded == -1)
    # for combined_mask_pos, should slightly increase the adv, the higher entropy, increase more adv
    if config.pos_alpha:
        advantages[combined_mask_pos] = advantages[combined_mask_pos] + config.pos_alpha * entropys[combined_mask_pos]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.neg_alpha:
        entropy_credit_neg = entropys.max(dim=1, keepdim=True).values - entropys
        advantages[combined_mask_neg] = advantages[combined_mask_neg] - config.neg_alpha * entropy_credit_neg[combined_mask_neg]
    advantages[(yl_rewards_expanded == 1) & grpo_calculation_mask.bool()] = torch.clamp(advantages[(yl_rewards_expanded == 1) & grpo_calculation_mask.bool()], min=0)
    advantages[(yl_rewards_expanded == -1) & grpo_calculation_mask.bool()] = torch.clamp(advantages[(yl_rewards_expanded == -1) & grpo_calculation_mask.bool()], max=0)
    return advantages


def advantage_wrap_noaccmask_qwen3(data, advantages, entropys, config):
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    grpo_calculation_mask = data.batch["response_mask"]
    # all-wrong examples using cpo
    old_log_probs = data.batch["old_log_probs"]
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    ratio = torch.clamp(ratio, 1/ config.cpo_lambda, config.cpo_lambda)
    gw_yl_prob_clip_mask = torch.exp(gw_yl_log_probs) > 0.1
    old_prob_clip_mask = torch.exp(old_log_probs) > 0.1
    ratio_mask_pos = ratio >= config.cpo_lambda
    ratio_mask_neg = ratio <= 1 / config.cpo_lambda
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, ratio.size(-1))
    combined_mask_pos = ratio_mask_pos & data.batch["response_mask"].bool() & gw_yl_prob_clip_mask
    combined_mask_neg = ratio_mask_neg & data.batch["response_mask"].bool() & old_prob_clip_mask
    # for combined_mask_pos, should slightly increase the adv, the higher entropy, increase more adv
    if config.pos_alpha:
        advantages[combined_mask_pos] = advantages[combined_mask_pos] + config.pos_alpha * entropys[combined_mask_pos]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.neg_alpha:
        entropy_credit_neg = entropys.max(dim=1, keepdim=True).values - entropys
        advantages[ combined_mask_neg] = advantages[combined_mask_neg] - config.neg_alpha * entropy_credit_neg[combined_mask_neg]
    advantages[(yl_rewards_expanded == 1) & grpo_calculation_mask.bool()] = torch.clamp(advantages[(yl_rewards_expanded == 1) & grpo_calculation_mask.bool()], min=0)
    advantages[(yl_rewards_expanded == -1) & grpo_calculation_mask.bool()] = torch.clamp(advantages[(yl_rewards_expanded == -1) & grpo_calculation_mask.bool()], max=0)
    return advantages


def advantage_wrap_mi_qwen3(data, advantages, entropys, config):
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    grpo_calculation_mask = data.batch["response_mask"]
    # 计算mutual information
    mi = torch.exp(gw_yl_log_probs) * (gw_yl_log_probs - old_log_probs)
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, ratio.size(-1))
    pos_position = grpo_calculation_mask.bool() & (mi >= 0)
    neg_position = grpo_calculation_mask.bool() & (mi < 0)
    if config.pos_alpha:
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * mi[pos_position] * uncertainty[pos_position]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.neg_alpha:
        advantages[neg_position] = advantages[neg_position] + config.neg_alpha * mi[neg_position] * (1 - uncertainty[neg_position])
    advantages[(yl_rewards_expanded == 1) & grpo_calculation_mask.bool()] = torch.clamp(advantages[(yl_rewards_expanded == 1) & grpo_calculation_mask.bool()], min=0)
    advantages[(yl_rewards_expanded == -1) & grpo_calculation_mask.bool()] = torch.clamp(advantages[(yl_rewards_expanded == -1) & grpo_calculation_mask.bool()], max=0)
    return advantages


def advantage_wrap_mi_accmask_qwen3(data, advantages, entropys, config):
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    grpo_calculation_mask = data.batch["response_mask"]
    mi = torch.exp(gw_yl_log_probs) * (gw_yl_log_probs - old_log_probs)
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, ratio.size(-1))
    if config.pos_alpha:
        pos_position = grpo_calculation_mask.bool() & (mi >= 0) & (yl_rewards_expanded == 1)
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * mi[pos_position] * (1 + uncertainty[pos_position])
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.neg_alpha:
        neg_position = grpo_calculation_mask.bool() & (mi < 0) & (yl_rewards_expanded == -1)
        advantages[neg_position] = advantages[neg_position] + config.neg_alpha * mi[neg_position] * (2 - uncertainty[neg_position])
    advantages[(yl_rewards_expanded == 1) & grpo_calculation_mask.bool()] = torch.clamp(advantages[(yl_rewards_expanded == 1) & grpo_calculation_mask.bool()], min=0)
    advantages[(yl_rewards_expanded == -1) & grpo_calculation_mask.bool()] = torch.clamp(advantages[(yl_rewards_expanded == -1) & grpo_calculation_mask.bool()], max=0)
    return advantages


def advantage_wrap_mi_unify_qwen3(data, advantages, entropys, config):
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    grpo_calculation_mask = data.batch["response_mask"]
    mi = torch.exp(gw_yl_log_probs) * (gw_yl_log_probs - old_log_probs)
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, ratio.size(-1))
    pos_position = grpo_calculation_mask.bool() &  (yl_rewards_expanded == 1)
    neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == -1)
    if config.pos_alpha:
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * mi[pos_position] * (1 - uncertainty[pos_position])
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.neg_alpha:
        advantages[neg_position] = advantages[neg_position] + config.neg_alpha * mi[neg_position] * (1 - uncertainty[neg_position])
    advantages[pos_position] = torch.clamp(advantages[pos_position], min=0)
    advantages[neg_position] = torch.clamp(advantages[neg_position], max=0)
    return advantages


def advantage_wrap_mi_clamp_unify_qwen3(data, advantages, entropys, config):
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    grpo_calculation_mask = data.batch["response_mask"]
    mi = torch.exp(gw_yl_log_probs) * (gw_yl_log_probs - old_log_probs)
    mi = torch.clamp(mi, max=10)
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, ratio.size(-1))
    pos_position = grpo_calculation_mask.bool() &  (yl_rewards_expanded == 1)
    neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == -1)
    if config.pos_alpha:
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * mi[pos_position] * (1 - uncertainty[pos_position])
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.neg_alpha:
        advantages[neg_position] = advantages[neg_position] + config.neg_alpha * mi[neg_position] * (1 - uncertainty[neg_position])
    advantages[pos_position] = torch.clamp(advantages[pos_position], min=0)
    advantages[neg_position] = torch.clamp(advantages[neg_position], max=0)
    return advantages


def advantage_wrap_mi_clamp_unify_difficulty_qwen3(data, advantages, entropys, config):
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    difficulty_mask = data.batch["difficulty_mask"]
    grpo_calculation_mask = data.batch["response_mask"]
    mi = torch.exp(gw_yl_log_probs) * (gw_yl_log_probs - old_log_probs)
    mi = torch.clamp(mi, max=10)
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    difficulty_mask_expanded = difficulty_mask.unsqueeze(-1).expand(-1, ratio.size(-1))
    pos_position = grpo_calculation_mask.bool() &  (difficulty_mask_expanded != 0) ## 全对或全错
    neg_position = grpo_calculation_mask.bool() & (difficulty_mask_expanded == 0) ## 有对有错
    if config.pos_alpha:
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * mi[pos_position] * (1 - uncertainty[pos_position])
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.neg_alpha:
        advantages[neg_position] = advantages[neg_position] + config.neg_alpha * mi[neg_position] * (1 - uncertainty[neg_position])
    return advantages


def advantage_wrap_mi3_qwen3(data, advantages, entropys, config):
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    if "yw_rewards" in data.batch:
        yw_rewards = data.batch["yw_rewards"]
        yw_rewards_expanded = yw_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    grpo_calculation_mask = data.batch["response_mask"]
    mi = torch.exp(gw_yl_log_probs) * (gw_yl_log_probs - old_log_probs)
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo
    pos_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1)
    neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1)
    if "yw_rewards" in data.batch:
        neg_position = neg_position & (yw_rewards_expanded == 1)
    # breakpoint()
    if config.get("pos_alpha") and config.pos_alpha:
        # for correct examples, enlarge its advantage according to uncertainty 
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * uncertainty[pos_position]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.get("neg_alpha") and config.neg_alpha:
        # for wrong examples, enlarge its advantage according to uncertainty and mi
        advantages[neg_position & (mi >= 0)] = advantages[neg_position & (mi >= 0)] + config.neg_alpha * mi[neg_position & (mi >= 0)] * uncertainty[neg_position & (mi >= 0)]
        advantages[neg_position & (mi < 0)] = advantages[neg_position & (mi < 0)] + config.neg_alpha * mi[neg_position & (mi < 0)] * (1 - uncertainty[neg_position & (mi < 0)]) 
    return advantages


def advantage_wrap_negonly_mi3_qwen3(data, advantages, entropys, config):
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    difficulty_mask = data.batch["difficulty_mask"]
    difficulty_mask_expanded = difficulty_mask.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    if "yw_rewards" in data.batch:
        yw_rewards = data.batch["yw_rewards"]
        yw_rewards_expanded = yw_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    grpo_calculation_mask = data.batch["response_mask"]
    mi = torch.exp(gw_yl_log_probs) * (gw_yl_log_probs - old_log_probs)
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo
    pos_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1)
    neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1) & (difficulty_mask_expanded == -1)
    if "yw_rewards" in data.batch:
        neg_position = neg_position & (yw_rewards_expanded == 1)
    if config.get("pos_alpha") and config.pos_alpha:
        # for correct examples, enlarge its advantage according to uncertainty 
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * uncertainty[pos_position]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.get("neg_alpha") and config.neg_alpha:
        # for wrong examples, enlarge its advantage according to uncertainty and mi
        advantages[neg_position & (mi >= 0)] = advantages[neg_position & (mi >= 0)] + config.neg_alpha * mi[neg_position & (mi >= 0)] * uncertainty[neg_position & (mi >= 0)]
        advantages[neg_position & (mi < 0)] = advantages[neg_position & (mi < 0)] + config.neg_alpha * mi[neg_position & (mi < 0)] * (1 - uncertainty[neg_position & (mi < 0)]) 
    return advantages


def advantage_wrap_negonly_mi_qwen3(data, advantages, entropys, config):
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    difficulty_mask = data.batch["difficulty_mask"]
    difficulty_mask_expanded = difficulty_mask.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    if "yw_rewards" in data.batch:
        yw_rewards = data.batch["yw_rewards"]
        yw_rewards_expanded = yw_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    grpo_calculation_mask = data.batch["response_mask"]
    mi = torch.exp(gw_yl_log_probs) * (gw_yl_log_probs - old_log_probs)
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo    
    pos_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1)
    neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1) & (difficulty_mask_expanded == -1)
    if "yw_rewards" in data.batch:
        neg_position = neg_position & (yw_rewards_expanded == 1)
    if config.get("pos_alpha") and config.pos_alpha:
        # for correct examples, enlarge its advantage according to uncertainty 
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * uncertainty[pos_position]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.get("neg_alpha") and config.neg_alpha:
        # for wrong examples, enlarge its advantage according to uncertainty and mi
        advantages[neg_position] = advantages[neg_position ] + config.neg_alpha * mi[neg_position ]
    return advantages


def advantage_wrap_reverse_mi3_qwen3(data, advantages, entropys, config):
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    grpo_calculation_mask = data.batch["response_mask"]
    mi = torch.exp(gw_yl_log_probs) * (gw_yl_log_probs - old_log_probs)
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, ratio.size(-1))
    pos_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1)
    neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1)
    if config.get("pos_alpha") and config.pos_alpha:
        # for correct examples, enlarge its advantage according to uncertainty 
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * uncertainty[pos_position]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.get("neg_alpha") and config.neg_alpha:
        # for wrong examples, enlarge its advantage according to uncertainty and mi
        advantages[neg_position & (mi >= 0)] = advantages[neg_position & (mi >= 0)] - config.neg_alpha * mi[neg_position & (mi >= 0)] * uncertainty[neg_position & (mi >= 0)]
        advantages[neg_position & (mi < 0)] = advantages[neg_position & (mi < 0)] - config.neg_alpha * mi[neg_position & (mi < 0)] * (1 - uncertainty[neg_position & (mi < 0)]) 
    return advantages


def advantage_wrap_addition_mi3_qwen3(data, advantages, entropys, config):
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    grpo_calculation_mask = data.batch["response_mask"]
    mi = torch.exp(gw_yl_log_probs) * (gw_yl_log_probs - old_log_probs)
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, ratio.size(-1))
    pos_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1)
    neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1)
    if config.get("pos_alpha") and config.pos_alpha:
        # for correct examples, enlarge its advantage according to uncertainty 
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * uncertainty[pos_position]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.get("neg_alpha") and config.neg_alpha:
        # for wrong examples, enlarge its advantage according to uncertainty and mi
        advantages[neg_position & (mi >= 0)] = advantages[neg_position & (mi >= 0)] + config.neg_alpha * mi[neg_position & (mi >= 0)] * uncertainty[neg_position & (mi >= 0)]
    return advantages


def advantage_wrap_mi2_qwen3(data, advantages, entropys, config):
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    grpo_calculation_mask = data.batch["response_mask"]
    diff = gw_yl_log_probs - old_log_probs
    mi = torch.exp(gw_yl_log_probs) * diff
    # 创建一个临时的 entropy，将 mask 位置设为极值
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # all-wrong examples using cpo
    ratio = torch.exp(gw_yl_log_probs - old_log_probs) 
    # breakpoint()
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, ratio.size(-1))
    pos_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1)
    neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1)
    if config.get("pos_alpha") and config.pos_alpha:
        # for correct examples, enlarge its advantage according to uncertainty 
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * uncertainty[pos_position]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.get("neg_alpha") and config.neg_alpha:
        # for wrong examples, enlarge its advantage according to uncertainty and mi
        advantages[neg_position] = advantages[neg_position] + config.neg_alpha * mi[neg_position]
    return advantages, mi


def advantage_wrap_negonly_seq_kl_qwen3(data, advantages, entropys, config):
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    difficulty_mask = data.batch["difficulty_mask"]
    difficulty_mask_expanded = difficulty_mask.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    if "yw_rewards" in data.batch:
        yw_rewards = data.batch["yw_rewards"]
        yw_rewards_expanded = yw_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    grpo_calculation_mask = data.batch["response_mask"]

    kl = (gw_yl_log_probs - old_log_probs)
    masked_kl = kl * grpo_calculation_mask
    sum_kl = masked_kl.sum(dim=1)  # 在 batch 维度求和
    count = grpo_calculation_mask.sum(dim=1)  # 统计每个位置有多少个有效样本
    kl_mean = sum_kl / (count + 1e-10)
    kl_mean_expanded = kl_mean.unsqueeze(-1).expand(-1, advantages.size(-1))
    
    pos_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1)
    neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1) & (difficulty_mask_expanded == -1)
    if "yw_rewards" in data.batch:
        neg_position = neg_position & (yw_rewards_expanded == 1)
    if config.get("pos_alpha") and config.pos_alpha:
        # for correct examples, enlarge its advantage according to uncertainty 
        advantages[pos_position] = advantages[pos_position] + config.pos_alpha * kl_mean_expanded[pos_position]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.get("neg_alpha") and config.neg_alpha:
        # for wrong examples, enlarge its advantage according to uncertainty and mi
        advantages[neg_position ] = advantages[neg_position] + config.neg_alpha * kl_mean_expanded[neg_position]
    return advantages


def advantage_wrap_seq_kl_qwen3(data, advantages, entropys, config):
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    difficulty_mask = data.batch["difficulty_mask"]
    difficulty_mask_expanded = difficulty_mask.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    if "yw_rewards" in data.batch:
        yw_rewards = data.batch["yw_rewards"]
        yw_rewards_expanded = yw_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    grpo_calculation_mask = data.batch["response_mask"]
    kl = (gw_yl_log_probs - old_log_probs)
    masked_kl = kl * grpo_calculation_mask
    sum_kl = masked_kl.sum(dim=1)  # 在 batch 维度求和
    count = grpo_calculation_mask.sum(dim=1)  # 统计每个位置有多少个有效样本
    kl_mean = sum_kl / (count + 1e-10)
    kl_mean_expanded = kl_mean.unsqueeze(-1).expand(-1, advantages.size(-1))
    

    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), 0)  # 用于计算 mean
    count_masked = grpo_calculation_mask.sum(dim=1)  # 统计每个位置有多少个有效样本
    uncertainty = entropys_masked.sum(dim=1) / (count_masked + 1e-10)
    # 归一化
    uncertainty = uncertainty.unsqueeze(-1).expand(-1, advantages.size(-1))

    pos_position_posonly = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1) & (difficulty_mask_expanded == 1)
    pos_position_pos = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1) & (difficulty_mask_expanded != 1)
    # neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1) 
    neg_position_negonly = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1) & (difficulty_mask_expanded == -1)
    neg_position_neg = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1) & (difficulty_mask_expanded != -1)
    # breakpoint()
    if "yw_rewards" in data.batch:
        neg_position = neg_position & (yw_rewards_expanded == 1)
    if config.get("pos_alpha") and config.pos_alpha:
        # for correct examples, enlarge its advantage according to uncertainty 
        advantages[pos_position_posonly] = advantages[pos_position_posonly] + config.pos_alpha * uncertainty[pos_position_posonly]
        advantages[pos_position_pos] = advantages[pos_position_pos] + torch.max(config.pos_alpha * uncertainty[pos_position_pos], -advantages[pos_position_pos].abs()/2)
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.get("neg_alpha") and config.neg_alpha:
        # for wrong examples, enlarge its advantage according to uncertainty and mi
        advantages[neg_position_negonly] = advantages[neg_position_negonly] + config.neg_alpha * kl_mean_expanded[neg_position_negonly]
        advantages[neg_position_neg] = advantages[neg_position_neg] + torch.min(config.neg_alpha * kl_mean_expanded[neg_position_neg], advantages[neg_position_neg].abs()/2) 
    return advantages


def advantage_wrap_negonly_tok_kl_qwen3(data, advantages, entropys, config):
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    difficulty_mask = data.batch["difficulty_mask"]
    difficulty_mask_expanded = difficulty_mask.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    if "yw_rewards" in data.batch:
        yw_rewards = data.batch["yw_rewards"]
        yw_rewards_expanded = yw_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    grpo_calculation_mask = data.batch["response_mask"]
    kl = (gw_yl_log_probs - old_log_probs)
    kl = torch.clamp(kl, min=-10, max=10)

    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零

    pos_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1)
    neg_position = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1) & (difficulty_mask_expanded == -1)
    if "yw_rewards" in data.batch:
        neg_position = neg_position & (yw_rewards_expanded == 1)
    if config.get("pos_alpha") and config.pos_alpha:
        # for correct examples, enlarge its advantage according to uncertainty 
        advantages[ pos_position] = advantages[pos_position] + config.pos_alpha * uncertainty[pos_position]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.get("neg_alpha") and config.neg_alpha:
        # for wrong examples, enlarge its advantage according to uncertainty and mi
        advantages[neg_position] = advantages[neg_position] + config.neg_alpha * kl[neg_position]
    return advantages

def advantage_wrap_tok_kl_qwen3(data, advantages, entropys, config):
    gw_yl_log_probs = data.batch["gw_yl_log_probs"]
    old_log_probs = data.batch["old_log_probs"]
    difficulty_mask = data.batch["difficulty_mask"]
    difficulty_mask_expanded = difficulty_mask.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    yl_rewards = data.batch["token_level_rewards"].sum(-1)
    yl_rewards_expanded = yl_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    if "yw_rewards" in data.batch:
        yw_rewards = data.batch["yw_rewards"]
        yw_rewards_expanded = yw_rewards.unsqueeze(-1).expand(-1, gw_yl_log_probs.size(-1))
    grpo_calculation_mask = data.batch["response_mask"]
    kl = (gw_yl_log_probs - old_log_probs)
    kl = torch.clamp(kl, min=-10, max=10)


    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('inf'))  # 用于计算 min
    min_vals = entropys_masked.min(dim=1, keepdim=True).values
    entropys_masked = entropys.clone()
    entropys_masked = entropys_masked.masked_fill(~grpo_calculation_mask.bool(), float('-inf'))  # 用于计算 max
    max_vals = entropys_masked.max(dim=1, keepdim=True).values
    # 归一化
    uncertainty = (entropys - min_vals) / (max_vals - min_vals + 1e-8)  # 加小值防止除零
    # breakpoint()

    pos_position = grpo_calculation_mask.bool() & (yl_rewards_expanded == 1)
    neg_position_negonly = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1) & (difficulty_mask_expanded == -1)
    neg_position_neg = grpo_calculation_mask.bool() & (yl_rewards_expanded != 1) & (difficulty_mask_expanded != -1)
    if "yw_rewards" in data.batch:
        neg_position = neg_position & (yw_rewards_expanded == 1)
    if config.get("pos_alpha") and config.pos_alpha:
        # for correct examples, enlarge its advantage according to uncertainty 
        advantages[ pos_position] = advantages[pos_position] + config.pos_alpha * uncertainty[pos_position]
    # for combined_mask_neg, should slightly decrease the adv, the higher entropy, decrease less adv
    if config.get("neg_alpha") and config.neg_alpha:
        # for wrong examples, enlarge its advantage according to uncertainty and mi
        advantages[neg_position_negonly] = advantages[neg_position_negonly ] + config.neg_alpha * kl[neg_position_negonly]
        advantages[neg_position_neg] = advantages[neg_position_neg] +  torch.min(config.neg_alpha * kl[neg_position_neg], advantages[neg_position_neg].abs()/2) 
    return advantages