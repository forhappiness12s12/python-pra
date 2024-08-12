from transformers import pipeline

# Initialize the sentiment classifier pipeline with a pre-trained model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define the input sentences
sentences = [
    "I need help ",
    "I don't know the method of use.",
    "I lose my many money, I need urgent help.",
    "I have emergency,please help me as soon as ppssible"
]

def determine_urgency(sentence):
    # Define urgency keywords and their corresponding levels
    urgency_keywords = {
        "urgent": 3,
        "as soon as possible": 3,
        "emergency": 3,
        "need help": 2,
        "important": 2,
        "do not know": 1,
        "not sure": 1,
        "how to use": 1,
    }
    
    # Check for keywords in the sentence
    for keyword, level in urgency_keywords.items():
        if keyword in sentence.lower():
            return level
    return 1  # Default to level 1 if no keywords match

def assign_urgency(sentences):
    urgency_levels = [determine_urgency(sentence) for sentence in sentences]
    
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
print("Prioritized order:", prioritized_order)
