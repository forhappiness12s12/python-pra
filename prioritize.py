import torch
from transformers import pipeline

# Initialize the classifier pipeline with a pre-trained model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define the input sentences
sentences = [
    "I need help as soon as possible.",
    "I don't know the method of use.",
    "I lose my many money, I need urgent help."
]

# Define a custom function to assign urgency levels
def assign_urgency(sentences):
    urgency_map = {
        "LABEL_0": 1,  # Least urgent
        "LABEL_1": 2,  # More urgent
        "LABEL_2": 3   # Most urgent
    }

    results = classifier(sentences)
    urgency_levels = [urgency_map[result['label']] for result in results]
    
    # Pair each sentence with its urgency level
    paired = list(zip(sentences, urgency_levels))
    
    # Sort sentences by urgency level
    paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)
    
    # Extract and return the sorted sentences and their original indexes
    sorted_sentences = [s[0] for s in paired_sorted]
    original_indexes = [sentences.index(s[0]) + 1 for s in paired_sorted]
    return original_indexes

# Get the prioritized order of sentences
prioritized_order = assign_urgency(sentences)

# Output the prioritized order
print(prioritized_order)
