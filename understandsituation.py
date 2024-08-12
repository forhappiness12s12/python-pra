import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # num_labels depends on your classification task

# Initialize the pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define the text to classify
texts = [
    "negative.",
    "This is a negative statement."
]

# Classify the text
results = classifier(texts)

# Print the results
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {result['label']}, Confidence: {result['score']:.4f}\n")
