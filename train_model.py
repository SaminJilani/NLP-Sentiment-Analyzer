import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from datasets import Dataset
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# Load IMDB dataset
imdb_df = pd.read_csv('IMDB Dataset.csv')
imdb_df['sentiment'] = imdb_df['sentiment'].map({'positive': 1, 'negative': 0, 'neutral': -1})

# Drop rows with missing/NaN values in 'review' or 'sentiment' columns
imdb_df.dropna(subset=['review', 'sentiment'], inplace=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(imdb_df['review'], imdb_df['sentiment'], test_size=0.2,
                                                    random_state=42)

# Load pre-trained RoBERTa tokenizer from transformers module
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# Tokenize and encode the text data
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

def encode_labels(labels):
    return [2 if label == 1 else 1 if label == -1 else 0 for label in labels]

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': encode_labels(y_train.tolist())
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': encode_labels(y_test.tolist())
})


# Calculate class weights to handle class imbalance
class_counts = pd.Series(imdb_df['sentiment']).value_counts().sort_index()
class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[-1], 1.0 / class_counts[1]]).to(torch.float32)
class_weights = class_weights / class_weights.sum() * 3  # Normalize weights to sum to 3

# Path to the checkpoint
checkpoint_path = './roberta_results/checkpoint-5500'
# Check if the checkpoint exists
if os.path.exists(checkpoint_path):
    print("Checkpoint found. Resuming training from checkpoint.")
    model = RobertaForSequenceClassification.from_pretrained(checkpoint_path, num_labels=3)
else:
    print("No checkpoint found. Starting training from scratch.")
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./roberta_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./roberta_logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="steps",
    save_steps=500,
    resume_from_checkpoint=checkpoint_path if os.path.exists(checkpoint_path) else None
)


# Custom class for trainer to handle class weights and evaluation metrics
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_prediction: EvalPrediction):
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
        preds = logits.argmax(-1)

        # Calculate accuracy
        accuracy = accuracy_score(labels, preds)

        # Calculate precision, recall, F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# Initialize Trainer
trainer = WeightedTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset  # evaluation dataset
)

# Train the model
if os.path.exists(checkpoint_path):
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    trainer.train()

# Save the trained model
model.save_pretrained('roberta_imdb_sentiment_model')

# Evaluate the model on test set
evaluation_result = trainer.evaluate()

# Print evaluation metrics
print("Evaluation results:")
for key, value in evaluation_result.items():
    print(f"{key}: {value}")

print("Model training completed and saved as roberta_imdb_sentiment_model")
