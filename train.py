from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from ld import write_airo_ai_model
import fire
from data import get_dataset_cls
from metrics import get_metric_cls


def train(
        file_path: str,
        model_id: str,
        transformer: str = "distilbert/distilbert-base-uncased",
        learning_rate: float = 2e-5,
        epochs: int = 2,
        weight_decay: float = 0.01,
        problem_type: str = "single_label_classification"
):
    # First load utility classes for data and metrics
    dataset = get_dataset_cls(problem_type)(file_path)
    metrics = get_metric_cls(problem_type)

    # Then load tokenizer, model, data collator, training arguments, and trainer
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    tokenized_data = dataset.format().map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=model_id,
        learning_rate=learning_rate,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,  # we'll push manually at the end to grab the commitinfo
        push_to_hub_model_id=model_id
    )

    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        transformer,
        num_labels=len(dataset.id2label),
        id2label=dataset.id2label,
        label2id=dataset.label2id,
        problem_type=problem_type
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics().compute,
    )

    # Train
    trainer.train()

    # Push best model to hub and evaluate metrics
    model_url = trainer.push_to_hub(blocking=True)
    results = trainer.evaluate()

    # Generate LD metadata
    graph = write_airo_ai_model(model_id, model_url, results)

    with open("model-metadata.ttl", "w") as f:
        f.write(graph.serialize(format="turtle"))


if __name__ == "__main__":
    fire.Fire(train)