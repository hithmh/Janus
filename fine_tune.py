# nt.anh.fai@gmail.com

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from datasets import Dataset

# Định nghĩa model path
model_name = "Janus-Pro-7B"

# Load processor và tokenizer
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_name)
tokenizer = vl_chat_processor.tokenizer

# Load model
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# ---- Tạo dataset mẫu ----
dataset_samples = [
    {"question": "What is this image about?", "image": "images/image-283.png", "answer": "This image is about AI."},
    {"question": "Describe the object in the image.", "image": "images/image-283.png", "answer": "It is a red car."},
    {"question": "What can you infer from this?", "image": "images/image-283.png", "answer": "It seems like a festival."},
]

dataset = Dataset.from_dict({
    "question": [item["question"] for item in dataset_samples],
    "image": [item["image"] for item in dataset_samples],
    "answer": [item["answer"] for item in dataset_samples],
})


# ---- Chuẩn bị dữ liệu huấn luyện ----
def preprocess_function(examples):
    conversations = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{q}",
            "images": [img],
        }
        for q, img in zip(examples["question"], examples["image"])
    ]

    pil_images = load_pil_images(conversations)
    inputs = vl_chat_processor(
        conversations=conversations, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    labels = tokenizer(examples["answer"], padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**inputs)
    return {"inputs_embeds": inputs_embeds, "labels": labels}


tokenized_datasets = dataset.map(preprocess_function, batched=True)

# ---- Huấn luyện mô hình ----
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=vl_gpt,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

trainer.train()

# ---- Lưu mô hình sau fine-tune ----
vl_gpt.save_pretrained("./fine_tuned_Janus_Pro_7B")
tokenizer.save_pretrained("./fine_tuned_Janus_Pro_7B")

print("Fine-tuning complete!")
