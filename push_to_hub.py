from datasets import Dataset
from huggingface_hub import login
import json

login(token="api_key")

# For large JSONL files, process in chunks
def jsonl_generator(filepath):
    with open(filepath, 'r') as f:
        for line in f:
            yield json.loads(line)

dataset = Dataset.from_generator(
    lambda: jsonl_generator('<filename>.jsonl')
)

dataset.push_to_hub("<user_name>/<dataset_name>", private=False)