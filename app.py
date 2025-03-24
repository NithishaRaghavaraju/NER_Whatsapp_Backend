from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, pipeline, AutoTokenizer, AutoModelForTokenClassification
import re

app = Flask(__name__)
CORS(app)

# Load chatbot model
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Load POS tagging pipeline
pos_pipe = pipeline("token-classification", model="TweebankNLP/bertweet-tb2-pos-tagging")

# Load NER model
model_checkpoint = "huggingface-course/bert-finetuned-ner"
ner_model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)
ner_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
token_classifier = pipeline(
    "token-classification", model=ner_model, aggregation_strategy="simple", tokenizer=ner_tokenizer,
)

# Function to clean messages
def clean_message(text):
    # Remove emojis and special characters (except spaces and letters)
    text = re.sub(r'[^\w\s]', '', text)

    # Reduce repeated letters only if they appear more than twice at the end
    text = re.sub(r'(\w*?)(\w)\2{2,}\b', r'\1\2', text)

    # Perform POS tagging
    pos_tags = pos_pipe(text)

    # Convert words to title case selectively
    words = text.split()
    cleaned_words = []

    for i, word in enumerate(words):
        tag = next((tag_info["entity"] for tag_info in pos_tags if tag_info["word"] == word), None)

        if tag in ["ADJ", "ADP"]:  # Keep ADJ and ADP words lowercase
            cleaned_words.append(word.lower())
        else:  # Title case for other words
            cleaned_words.append(word.title())

    # Remove single-letter words (except 'I' or 'A' if needed)
    cleaned_words = [word for word in cleaned_words if len(word) > 1]

    return " ".join(cleaned_words)

# Function to extract named entities from a single message
def extract_entities(text, message_index, existing_entities=set(), threshold=0.85):
    entities_dict = {"PER": [], "ORG": [], "LOC": [], "MISC": []}
    seen_words = set(existing_entities)  # Initialize the set of previously noted entities

    results = token_classifier(text)

    for entity in results:
        word = entity["word"]
        entity_type = entity["entity_group"]
        score = entity["score"]

        # Ignore low-confidence entities
        if score < threshold:
            continue

        # Ignore subword tokens (split words like "##word")
        if word.startswith("##"):
            continue

        # Ignore short words (e.g., single letters)
        if len(word) == 1:
            continue

        # Keep multi-word locations intact
        if entity_type == "LOC":
            processed_words = [word]
        else:
            processed_words = word.split()

        for single_word in processed_words:
            # Check if the word has been already noted
            if single_word not in seen_words:
                seen_words.add(single_word)
                # Add new word to the respective entity list
                if entity_type in entities_dict:
                    entities_dict[entity_type].append({
                        "index": message_index,
                        "word": single_word,
                        "substring": (text.find(single_word), text.find(single_word) + len(single_word))
                    })

    return entities_dict





@app.route("/api/home", methods=['POST'])
def receive_message():
    data = request.get_json()
    message_index = data.get("index")
    message = data.get("message", "")

    print(f"Received message at index {message_index}: {message}")

    # Clean user message
    cleaned_message = clean_message(message)
    print("Cleaned Message:", cleaned_message)

    # Extract named entities from user message
    user_entities = extract_entities(cleaned_message, message_index)
    print("Extracted Entities from User's Message:", user_entities)

    # Generate chatbot response
    inputs = tokenizer(cleaned_message, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    bot_response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    print(f"Chatbot Response: {bot_response}")

    # The bot's response index will be the user message index + 1
    bot_index = message_index + 1

    # Extract named entities from chatbot response (bot index)
    bot_entities = extract_entities(bot_response, bot_index)
    print("Extracted Entities from Chatbot's Response:", bot_entities)

    return jsonify({
        'response': bot_response,
        'person_user': user_entities.get("PER", []),
        'location_user': user_entities.get("LOC", []),
        'person_bot': bot_entities.get("PER", []),
        'location_bot': bot_entities.get("LOC", [])
    })


if __name__ == "__main__":
    app.run(debug=True, port=8050, host='0.0.0.0')
