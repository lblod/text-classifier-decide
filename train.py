import json
from abc import abstractmethod, ABC

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import Dataset
from ld import write_airo_ai_model
import fire


def sigmoid(x):
   return 1/(1 + np.exp(-x))


class Metrics(ABC):

    @abstractmethod
    def map_predictions(self, predictions):
        pass

    def compute(self, eval_pred):
        preds, labels = eval_pred
        preds = self.map_predictions(preds)
        accuracy = accuracy_score(labels.astype(int), preds)

        # Calculate precision, recall, and F1-score
        precision = precision_score(labels.astype(int), preds, average='weighted')
        recall = recall_score(labels.astype(int), preds, average='weighted')
        f1 = f1_score(labels.astype(int), preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class SingleLabelMetrics(Metrics):
    def map_predictions(self, predictions):
        return np.argmax(predictions, axis=1).astype(int)


class MultiLabelMetrics(Metrics):
    def map_predictions(self, predictions):
        predictions = sigmoid(predictions)
        return (predictions > 0.5).astype(int)


def read_data(file_path: str):
    ## Todo this should straight from LD
    with open(file_path) as fd:
        data = json.loads(fd.read())
        print(len(data))

    return data


def format_data_single(data, label2id) -> Dataset:
    return Dataset.from_list([
        {
            "text": task["data"]["text"],
            "label": label2id[task["annotations"][0]["result"][0]["value"]["choices"][0]]
        }
        for task in data if task["annotations"][0]["result"]
    ])


def format_data_multi(data, label2id) -> Dataset:
    return Dataset.from_list([
        {
            "text": task["data"]["text"],
            "label": np.isin(
                np.arange(len(label2id)),
                [label2id[l] for l in task["annotations"][0]["result"][0]["value"]["choices"]]
            ).astype(float).tolist()
        }
        for task in data if task["annotations"][0]["result"]
    ])


def generate_label_map(data):
    label_set = {l for task in data if task["annotations"][0]["result"] for l in task["annotations"][0]["result"][0]["value"]["choices"]}
    id2label = {idx: label for idx, label in enumerate(label_set)}
    label2id = {label: idx for idx, label in enumerate(label_set)}
    return label2id, id2label


def train(
        file_path: str,
        model_id: str,
        transformer: str = "distilbert/distilbert-base-uncased",
        learning_rate: float = 2e-5,
        epochs: int = 2,
        weight_decay: float = 0.01,
        problem_type: str = "single_label_classification"
):
    option_map = {
        'single_label_classification': (SingleLabelMetrics, format_data_single),
        'multi_label_classification': (MultiLabelMetrics, format_data_multi)
    }

    metrics, format_data = option_map[problem_type]
    data = read_data(file_path)
    label2id, id2label = generate_label_map(data)
    dataset = format_data(data, label2id)

    tokenizer = AutoTokenizer.from_pretrained(transformer)

    if problem_type == 'single_label_classification':
        dataset = dataset.class_encode_column("label")
        dataset = dataset.train_test_split(test_size=0.1, stratify_by_column="label")
    else:
        dataset = dataset.train_test_split(test_size=0.1)

    tokenized_data = dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="uc1-uc2-model",
        learning_rate=learning_rate,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        push_to_hub_model_id=model_id
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        transformer,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        problem_type=problem_type
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics().compute,
    )


    trainer.train()
    model_url = trainer.push_to_hub(blocking=True)
    results = trainer.evaluate()
    graph = write_airo_ai_model(model_id, model_url, results)

    with open("model-metadata.ttl", "w") as f:
        f.write(graph.serialize(format="turtle"))


if __name__ == "__main__":
    fire.Fire(train)