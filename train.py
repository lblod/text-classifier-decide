import datetime
import json
import numpy as np
from functools import partial

import evaluate
import pytz
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import Dataset
from rdflib import Graph


class Metrics:
    def __init__(self):
        super().__init__()
        self.accuracy = evaluate.load("accuracy")

    def compute(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=labels)


def read_data(file_path: str):
    ## Todo this should straight from LD
    with open(file_path) as fd:
        data = json.loads(fd.read())
    return data


def format_data(data, label2id) -> Dataset:
    return Dataset.from_list([
        {"text": task["data"]["text"], "label": label2id[task["annotations"][0]["result"][0]["value"]["choices"][0]]}
        for task in data
    ])


def generate_label_map(data):
    label_set = {l for task in data for l in task["annotations"][0]["result"][0]["value"]["choices"]}
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
    tokenized_data = dataset.map(partial(tokenizer, truncation=True), batched=True)
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

    return {
        "dct:title": model_id,
        "sd:datePublished": datetime.datetime.now(tz=pytz.timezone("Europe/Brussels")).isoformat(),

    }

