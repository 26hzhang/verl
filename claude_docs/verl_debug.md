# VeRL框架

0. 下载模型和数据

```bash
huggingface-cli download --token YOUR_HF_TOKEN --repo-type dataset --resume-download BytedTsinghua-SIA/AIME-2024 --local-dir ~/Downloads/AIME-2024

huggingface-cli download --token YOUR_HF_TOKEN --resume-download font-info/qwen3-4b-sft-SGLang-RL --local-dir ~/Downloads/qwen3-4b-sft-SGLang-RL
```

1. verl/trainer/config/engine/fsdp.yaml中默认的model_dtype是fp32，Rayworker初始化模型的时候只会调用这个model_dtype，会出现：

```bash
Flash Attention 2 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen3Model is torch.float32....
```

修改方式：(1) 修改model_dtype为bf16；（2）在training scripts中添加actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16。

2. data.max_prompt_length + data.max_response_length < actor_rollout_ref.rollout.max_num_batched_tokens，否则会报错

| 参数名                     | 用途                          | 默认值 | 配置文件位置                                  |
| -------------------------- | ----------------------------- | ------ | --------------------------------------------- |
| `max_prompt_length`      | 单个prompt的最大token长度     | 512    | `verl/trainer/config/data/legacy_data.yaml` |
| `max_response_length`    | 单个response的最大token长度   | 512    | `verl/trainer/config/data/legacy_data.yaml` |
| `max_num_batched_tokens` | 批次中所有序列的token总数上限 | 8192   | `verl/trainer/config/rollout/rollout.yaml`  |

```yaml
# verl/trainer/config/data/legacy_data.yaml
max_prompt_length: 512
max_response_length: 512
# verl/trainer/config/rollout/rollout.yaml
# max number of tokens in a batch
max_num_batched_tokens: 8192
```

对应的数据类

```python
# verl/trainer/config/data_config.py
@dataclass
class DataConfig(BaseConfig):
    max_prompt_length: int = 512
    max_response_length: int = 512

# verl/workers/config/rollout.py
@dataclass
class RolloutConfig(BaseConfig):
    max_num_batched_tokens: int = 8192
```

细节参考：`docs/max_tokens_config_guide.md`

3. sglang在dataloader阶段做tokenizer，但是**在多轮**rollout阶段会重新做tokenizer，所以dataloader阶段的tokenizer结果会被丢弃
4. verl里vllm似乎不支持tools输入。sglang里的目前的方式对于不同轮数的数据处理也有bug：`verl/workers/rollout/sglang_rollout/sglang_rollout.py@1274：messages=raw_prompt.tolist() if isinstance(raw_prompt, np.ndarray) else raw_prompt`.
5. verl里如果当前数据里面每个instance的tools不一样，则会在缺失tools的instance里面把这个tool留一个None的占位符，这会导致后续报错：`verl/workers/rollout/sglang_rollout/sglang_rollout.py@1255：_tools_kwargs = {k: v for k, v in _tools_kwargs.items() if v is not None}`
6. verl新添加工具：

   1. 如果只想offline的用最后一轮工具的结果做RL，需要将 `actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1`，否则会调用tool去执行，并把tool执行结果拼回到message
   2. 实现一个tools.yaml，其中class_name中是具体的tool的执行代码的位置

   ```python
   tools:
       - class_name: examples.autoagent.project_runner_and_verifier_planner.project_runner_and_verifier_planner_tool.ProjectRunnerAndVerifierPlannerTool
         config:
           type: native
         tool_schema:
           function:
             description: Create a new plan with a specified goal and initialize its structure.
             name: create_plan
             parameters:
               additionalProperties: false
               properties:
                 goal:
                   description: goal
                   title: Goal
                   type: string
                 tasks:
                   description: tasks
                   items:
                     type: string
                   title: Tasks
                   type: array
               required:
               - goal
               - tasks
               type: object
           type: function
       - class_name: examples.autoagent.project_runner_and_verifier_planner.project_runner_and_verifier_planner_tool.ProjectRunnerAndVerifierPlannerTool
         config:
           type: native
         tool_schema:
           function:
             description: Advance to the next step by updating current step status based
               on execution results. Use this tool to skip completed or unnecessary steps
               efficiently. Can be called multiple times consecutively to skip several steps
               when appropriate.
             name: next_step
             parameters:
               additionalProperties: false
               properties:
                 status:
                   description: status
                   title: Status
                   type: boolean
               required:
               - status
               type: object
           type: function
       - class_name: examples.autoagent.project_runner_and_verifier_planner.project_runner_and_verifier_planner_tool.ProjectRunnerAndVerifierPlannerTool
         config:
           type: native
         tool_schema:
           function:
             description: Insert a new task into the plan at a specified position.
             name: insert_task
             parameters:
               additionalProperties: false
               properties:
                 task:
                   description: task
                   title: Task
                   type: string
                 task_index:
                   description: task_index
                   title: Task Index
                   type: integer
               required:
               - task
               - task_index
               type: object
           type: function
       - class_name: examples.autoagent.project_runner_and_verifier_planner.project_runner_and_verifier_planner_tool.ProjectRunnerAndVerifierPlannerTool
         config:
           type: native
         tool_schema:
           function:
             description: Create a new task with its initial steps within the plan.
             name: create_task
             parameters:
               additionalProperties: false
               properties:
                 steps:
                   description: steps
                   items:
                     type: string
                   title: Steps
                   type: array
                 task_index:
                   description: task_index
                   title: Task Index
                   type: integer
               required:
               - task_index
               - steps
               type: object
           type: function
       - class_name: examples.autoagent.project_runner_and_verifier_planner.project_runner_and_verifier_planner_tool.ProjectRunnerAndVerifierPlannerTool
         config:
           type: native
         tool_schema:
           function:
             description: Insert a new step into an existing task at a specified position.
             name: insert_step
             parameters:
               additionalProperties: false
               properties:
                 step:
                   description: step
                   title: Step
                   type: string
                 step_index:
                   description: step_index
                   title: Step Index
                   type: integer
               required:
               - step
               - step_index
               type: object
           type: function
       - class_name: examples.autoagent.project_runner_and_verifier_planner.project_runner_and_verifier_planner_tool.ProjectRunnerAndVerifierPlannerTool
         config:
           type: native
         tool_schema:
           function:
             description: Reset the current task and step indices to restart plan execution.
             name: reset_current_task
             parameters:
               additionalProperties: false
               properties:
                 step_id:
                   description: step_id
                   title: Step Id
                   type: integer
                 task_id:
                   description: task_id
                   title: Task Id
                   type: integer
               required:
               - task_id
               - step_id
               type: object
           type: function
       - class_name: examples.autoagent.project_runner_and_verifier_planner.project_runner_and_verifier_planner_tool.ProjectRunnerAndVerifierPlannerTool
         config:
           type: native
         tool_schema:
           function:
             description: '''\n        Transfer control to the executor with precise, context-aware
               instructions based on previous execution results. Include: \n            1)
               Specific step instruction adjusted based on prior outcomes (e.g., if a file
               was already created, instruct to modify it rather than create again),\n            2)
               Clear state of what has been accomplished vs. what remains to be done,\n            3)
               Concrete success criteria and expected outputs,\n            4) Previous execution
               results that impact the current step (e.g., generated file paths, API responses,
               error contexts),\n            5) Any constraints or considerations discovered
               during prior executions.\n        IMPORTANT: Instructions must be specific
               enough to prevent re-execution of completed work and build upon existing results.'''
             name: transfer_to_executor
             parameters:
               additionalProperties: false
               properties:
                 message:
                   description: message
                   title: Message
                   type: string
               required:
               - message
               type: object
           type: function
       - class_name: examples.autoagent.project_runner_and_verifier_planner.project_runner_and_verifier_planner_tool.ProjectRunnerAndVerifierPlannerTool
         config:
           type: native
         tool_schema:
           function:
             description: Retrieve the execution results of a previously completed step or
               task.
             name: execute_result
             parameters:
               additionalProperties: false
               properties:
                 step_id:
                   default: '-1'
                   description: step_id
                   title: Step Id
                   type: integer
                 task_id:
                   description: task_id
                   title: Task Id
                   type: integer
               required:
               - task_id
               type: object
           type: function
   ```
   3. 在数据中exra_info添加tool相关的字段：need_tools_kwargs, tools_kwargs

   ```python
   new_item = {
       "data_source": "autoagent_tool",
       "prompt": messages,
       "ability": "math",
       "reward_model": {"style": "rule", "ground_truth": solution},
       "extra_info": {
           "split": split,
           "index": idx,
           "answer": answer_raw,
           "question": messages[-1],
           "need_tools_kwargs": True,
           "tools_kwargs": {x['function']['name']: {'description': x['function']['description'], 'parameters': x['function']['parameters']} for x in tools},
       },
   }
   ```
7. verl添加新的reward：

   1. `verl/utils/reward_score/init.py`中添加reward的调用实现
   2. `verl/utils/reward_score/autoagent_tool.py`新添加具体的reward实现
8. verl的多轮对话：

   1. 上下文拼接：

      1. qwen3会输出一个空 `<think></think>`。`chat_template`对于thinking的处理会在input message里最后一个user之后的assistant role的地方加上。而verl的上下文拼接是把最新的对话input_ids直接拼接到之前的self.input_ids。这导致之前上下文的think会一直保留。目前可以设置 `actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True`，在每次多轮准备prompt的时候重新过一遍 `apply_chat_template`。但是在verl检查的时候仍然是会将 `apply_chat_template`和原始的 `input_ids`匹配，所以还是会报warning。
      2. 拼接assistant的消息时，只会拼接真正模型生成的部分，`<|im_start|>assistant`的loss会被mask

      ```python
      <think>

      </think>

      <tool_call>
      {"name": "create_plan", "arguments": {"goal": "Develop a weather application with location-based weather display, 5-day forecast, temperature unit switching (Celsius/Fahrenheit), weather icons, and city search functionality.", "tasks": [{"task": "Discover project structure and identify required dependencies", "steps": [{"instruction": "List all files in the working directory to identify existing project structure and dependencies.", "status": "Pending"}]}, {"task": "Analyze and parse dependency files to extract required packages", "steps": [{"instruction": "Search for dependency files such as package.json, requirements.txt, pyproject.toml, etc., and parse them to extract package names and version constraints.", "status": "Pending"}]}, {"task": "Install required dependencies for the weather application", "steps": [{"instruction": "Generate and execute installation scripts for Python (pip) or JavaScript (npm) based on the parsed dependency files.", "status": "Pending"}]}, {"task": "Set up the project structure with necessary directories and configuration files", "steps": [{"instruction": "Create directories for frontend, backend, and configuration files (e.g., config.json, routes.js, index.html).", "status": "Pending"}]}, {"task": "Implement core weather functionality", "steps": [{"instruction": "Develop the core weather service that fetches current weather and 5-day forecast based on user location or city name.", "status": "Pending"}]}, {"task": "Add temperature unit switching (Celsius/Fahrenheit)", "steps": [{"instruction": "Implement a toggle feature to switch between Celsius and Fahrenheit temperature units.", "status": "Pending"}]}, {"task": "Integrate weather icons", "steps": [{"instruction": "Fetch and display appropriate weather icons based on weather conditions (e.g., sunny, rainy, cloudy).", "status": "Pending"}]}, {"task": "Implement city search functionality", "steps": [{"instruction": "Allow users to search for a city by name and display its weather data.", "status": "Pending"}]}, {"task": "Run the application and verify functionality", "steps": [{"instruction": "Start the application server and verify that all features work correctly, including location-based weather, 5-day forecast, unit switching, icons, and city search.", "status": "Pending"}]}, {"task": "Test and debug the application", "steps": [{"instruction": "Test all features systematically, capture any errors or issues, and fix them.", "status": "Pending"}]}]}}
      </tool_call><|im_end|>
      ```
      3. 拼接user的消息时，会拼接开头和结尾的特殊字符，整个user的loss会被mask

      ```python
      <|im_start|>user
      <tool_response>
      Current parsed answer='' reward=0
      </tool_response><|im_end|>
      ```
      4. 在多轮rollout的时候，在新的生成之前，额外拼上 `<|im_start|>assistant`，并mask loss：`verl/workers/rollout/schemas.py@get_generation_prompt_ids`
