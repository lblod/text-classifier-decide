import json
import numpy as np
from functools import partial

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import Dataset
from ld import write_airo_ai_model
import fire


class Metrics:

    def compute(self, eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        accuracy = accuracy_score(labels, preds)

        # Calculate precision, recall, and F1-score
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


def read_data(file_path: str):
    ## Todo this should straight from LD
    with open(file_path) as fd:
        data = json.loads(fd.read())
        print(len(data))

    return data


def format_data(data, label2id) -> Dataset:
    return Dataset.from_list([
        {"text": task["data"]["text"], "label": label2id[task["annotations"][0]["result"][0]["value"]["choices"][0]]}
        for task in data if task["annotations"][0]["result"]
    ])


def generate_label_map(data):
    label_set = {l for task in data if task["annotations"][0]["result"] for l in task["annotations"][0]["result"][0]["value"]["choices"]}
    id2label = {idx: label for idx, label in enumerate(label_set)}
    label2id = {label: idx for idx, label in enumerate(label_set)}
    return label2id, id2label


def train(file_path: str, model_id: str):
    data = read_data(file_path)
    label2id, id2label = generate_label_map(data)
    dataset = format_data(data, label2id)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    metrics = Metrics()
    dataset = dataset.class_encode_column("label")
    dataset = dataset.train_test_split(test_size=0.1, stratify_by_column="label")
    tokenized_data = dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="uc1-uc2-model",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        push_to_hub_model_id=model_id
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics.compute,
    )


    trainer.train()
    model_url = trainer.push_to_hub(blocking=True)
    results = trainer.evaluate()
    graph = write_airo_ai_model(model_id, model_url, results)

    with open("model-metadata.ttl", "w") as f:
        f.write(graph.serialize(format="turtle"))


if __name__ == "__main__":
    fire.Fire(train)