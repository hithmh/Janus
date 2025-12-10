from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

base_model_name = "EleutherAI/pythia-1b-deduped"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
if tokenizer.chat_template is None:
    tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
reward_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)
ref_policy = AutoModelForCausalLM.from_pretrained(base_model_name)
policy = AutoModelForCausalLM.from_pretrained(base_model_name)

train_dataset = ... # make sure to have columns "input_ids"
eval_dataset = ...

trainer = RLOOTrainer(
    config=RLOOConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        total_episodes=30000,
    ),
    tokenizer=tokenizer,
    policy=policy,
    ref_policy=ref_policy,
    reward_model=reward_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()