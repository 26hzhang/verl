# CPO (Constrained Policy Optimization) Implementation Details

## Overview
CPO extends GRPO by introducing hint-based advantage adjustment for improved training on reasoning tasks (especially math).

## Key Files
- `verl/cpo/cpo_advantage_wrapper.py` - 17 advantage wrapper functions
- `verl/cpo/cpo_utils.py` - Utility functions (hint injection, difficulty masks, token analysis)
- `verl/trainer/ppo/ray_trainer.py` - CPO integration in trainer (lines 63-67, 1176-1202)
- `verl/workers/actor/dp_actor.py` - Actor-side CPO logic (lines 42-61, 390-507)
- `verl/trainer/ppo/core_algos.py` - CPO registered as AdvantageEstimator (line 108)

## Core Parameters

### cpo_lambda
- **Purpose**: Ratio clipping parameter to prevent policy divergence
- **Usage**: `clamped_ratio = torch.clamp(ratio, 1.0/cpo_lambda, cpo_lambda)`
- **Typical Range**: 1.5 - 5.0
  - Lower (1.5-2.5): Conservative, prevents large policy swings
  - Higher (5.0): More lenient, allows larger updates
- **Configuration**: Must be set at both algorithm and actor level
  ```bash
  +algorithm.cpo_lambda=${cpo_lambda}
  +actor_rollout_ref.actor.cpo_lambda=${cpo_lambda}
  ```

### Other CPO Parameters
- **pos_alpha**: Scaling for positive examples (correct answers)
- **neg_alpha**: Scaling for negative examples (incorrect answers)
- **gold_as_hint**: Whether to use ground truth as hints
- **wrap_method**: Which advantage wrapper to use (17 options)

## Advantage Wrapper Methods (17 total)
1. `tok_kl` - Token-level KL divergence
2. `negonly_mi` - Negative samples only, mutual information
3. `negonly_tok_kl` - Negative samples only, token KL
4. `seq_kl` - Sequence-level KL (commonly used)
5. `negonly_seq_kl` - Negative samples only, sequence KL
6. `negonly_mi3` - Negative samples only, MI variant 3
7. `addition_mi3` - Addition-based MI variant 3
8. `reverse_mi3` - Reverse-based MI variant 3
9. `mi` - Standard mutual information
10. `mi2` - MI variant 2
11. `mi3` - MI variant 3
12. `mi_clamp_unify_difficulty` - MI with difficulty clamping
13. `mi_clamp_unify` - MI with clamping
14. `mi_unify` - Unified MI approach
15. `noaccmask` - No accuracy mask
16. `mi_accmask` - MI with accuracy mask
17. `naive_qwen3` - Default for Qwen3 models (fallback)

## CPO Training Flow
```
1. Rollout Phase
   ├─ Generate responses from current policy
   └─ Compute token-level rewards

2. CPO-Specific Computation (in ray_trainer.py)
   ├─ Compute difficulty_mask: -1 (all wrong), 0 (mixed), 1 (all correct)
   ├─ Extract ground-truth hints from extra_info
   ├─ Inject hints into input sequences
   ├─ Recompute log probabilities with hints (gw_yl_log_probs)
   └─ Calculate mutual information (MI)

3. Advantage Adjustment (in dp_actor.py)
   ├─ Apply selected advantage wrapper
   ├─ Modify advantages based on:
   │  ├─ Mutual information (MI)
   │  ├─ Token-level uncertainty (entropy)
   │  ├─ Difficulty mask
   │  ├─ KL divergence
   │  └─ cpo_lambda clipping
   └─ Result: fine-grained advantage modifications

4. Policy Update
   └─ Standard PPO update with modified advantages
```

## Activation Requirements
CPO is activated when:
- `algorithm.adv_estimator = 'cpo'` is set
- Ground-truth data is present in batch (`extra_info` field with solutions)
- Reward computation is enabled

## Example Configuration (from run_qwen3_base_math_grpo_cpo_seqkl_neg.sh)
```bash
cpo_lambda=2              # Conservative clipping
pos_alpha=0               # No positive example adjustment
neg_alpha=1               # Full negative example adjustment

algorithm.adv_estimator=cpo
+algorithm.cpo_lambda=${cpo_lambda}
+algorithm.pos_alpha=${pos_alpha}
+algorithm.neg_alpha=${neg_alpha}
+algorithm.gold_as_hint=true
+algorithm.wrap_method="seq_kl"

+actor_rollout_ref.actor.cpo_lambda=${cpo_lambda}
+actor_rollout_ref.actor.pos_alpha=${pos_alpha}
+actor_rollout_ref.actor.neg_alpha=${neg_alpha}
+actor_rollout_ref.actor.wrap_method="seq_kl"
```

## Integration Points in Code
- Check for CPO mode: `if "gw_yl_log_probs" in data.batch.keys()`
- Wrapper selection: Based on `self.config.get("wrap_method")`
- All wrappers imported in dp_actor.py from cpo_advantage_wrapper module
