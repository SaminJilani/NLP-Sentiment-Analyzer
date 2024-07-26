import string

from flask import Flask, request, jsonify, render_template
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import nltk
from nltk import word_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from helperFunctions import contractions_dict, analyze_sentiment_with_context



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('wordnet')

app = Flask(__name__)

# Load the trained RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta_imdb_sentiment_model')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data_preprocessing_of_the_review(data.get('review', ''))
    print(review)
    if len(word_tokenize(review)) < 5:
        return jsonify({'error': 'Review must contain at least 10 meaningful words.'}), 400

    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    sentiment, confidence, key_phrases = analyze_sentiment(review)
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence,
        'key_phrases': key_phrases
    })


def predict_with_confidence(model, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    sentiment_index = torch.argmax(probabilities).item()
    confidence = probabilities[sentiment_index].item() * 100

    # Initialize sentiment list according to the correct order
    sentiments = ['Negative', 'Neutral', 'Positive']

    # Set the sentiment based on the model output and confidence
    sentiment = sentiments[sentiment_index]

    # Handle case where confidence is below threshold

    confidence = round(confidence, 2)
    return sentiment, confidence

def analyze_sentiment(review, confidence_threshold=65):
    # Perform initial sentiment prediction with the model
    final_sentiment, model_confidence = predict_with_confidence(model, review)
    # Apply verification to adjust the sentiment if necessary
    verified_sentiment, verified_confidence = analyze_sentiment_with_context(review)
    # Calculate degree of uncertainty
    verified_uncertainty = 100 - verified_confidence
    # Decide final sentiment based on verification and confidence
    if final_sentiment != 'Neutral':
        if verified_uncertainty < 50:
            if verified_confidence >= confidence_threshold:
                final_sentiment = verified_sentiment
                mean_confidence = (model_confidence + verified_confidence) / 2
                model_confidence = mean_confidence
        else:
            if verified_confidence > 30 and verified_confidence < 50:
                mean_confidence = (model_confidence + verified_confidence) / 2
                model_confidence = mean_confidence
        if verified_sentiment == 'Neutral':
            final_sentiment = verified_sentiment

    else:
        if model_confidence > 80:
            if verified_sentiment == 'Neutral':
                final_sentiment = 'Neutral'
            else:
                if verified_confidence > 65:
                    final_sentiment = verified_sentiment
                else:
                    final_sentiment = 'Negative'

    # Extract key phrases based on the final sentiment
    if model_confidence <= confidence_threshold:
        final_sentiment = 'Neutral'
    key_phrases = extract_key_phrases(review, final_sentiment)

    return final_sentiment, model_confidence, key_phrases


def extract_key_phrases(review, sentiment="both"):
    tokens = word_tokenize(review)
    pos_tags = pos_tag(tokens)
    lexicon = SentimentIntensityAnalyzer()
    negation_words = ["not", "no", "never", "n't", "ain't","didn't"]
    negative_phrases_list = ["not meet", "didn't meet", "not sure"]

    positive_phrases = set()
    negative_phrases = set()
    neutral_phrases = set()
    is_negated = False

    # Detect specific negative phrases
    for i in range(len(tokens) - 1):
        phrase_to_check = ' '.join(tokens[i:i + 2]).lower()
        if phrase_to_check in negative_phrases_list:
            negative_phrases.add(phrase_to_check)
            # Skip the next token since it's part of a recognized phrase
            continue

    for i, (word, tag) in enumerate(pos_tags):
        lower_word = word.lower()

        if lower_word in negation_words:
            is_negated = True
            continue

        if tag in ('NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RB') and lower_word not in stopwords.words('english'):
            sentiment_scores = lexicon.polarity_scores(word)
            if sentiment_scores:
                compound_score = sentiment_scores["compound"]
                sentiment_text = "neutral"
                if compound_score > 0.2:
                    sentiment_text = "Positive"
                elif compound_score < -0.2:
                    sentiment_text = "Negative"

                if is_negated:
                    sentiment_text = "Negative" if sentiment_text == "Positive" else "Positive"
                    is_negated = False

                if sentiment_text == "Positive":
                    positive_phrases.add(word)
                elif sentiment_text == "Negative":
                    negative_phrases.add(word)
                else:
                    neutral_phrases.add(word)

    # Convert sets to lists and remove any unwanted whitespace
    positive_phrases = [phrase for phrase in positive_phrases if phrase.strip()]
    negative_phrases = [phrase for phrase in negative_phrases if phrase.strip()]
    neutral_phrases = [phrase for phrase in neutral_phrases if phrase.strip()]

    print(f"positive_phrases: {positive_phrases}")
    print(f"negative_phrases: {negative_phrases}")
    print(f"neutral_phrases: {neutral_phrases}")
    ambiguity = ["Ambiguity Detected"]
    if sentiment == "Positive":
        if not positive_phrases:
            return ambiguity
        return positive_phrases
    elif sentiment == "Negative":
        if not negative_phrases:
            return ambiguity
        return negative_phrases
    elif sentiment == "Neutral":
        if not neutral_phrases:
            return ambiguity
        return neutral_phrases
    else:
        raise ValueError("Invalid sentiment parameter. Must be 'Positive', 'Negative', 'Neutral', or 'both'.")

def expand_contractions(text, contractions_dict):
    try:
        contractions_pattern = re.compile('({})'.format('|'.join(re.escape(key) for key in contractions_dict.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            expanded_contraction = contractions_dict.get(match.lower())
            if expanded_contraction:
                return expanded_contraction
            else:
                return match

        expanded_text = contractions_pattern.sub(expand_match, text)
        return expanded_text
    except Exception as e:
        print(f"Error in expanding contractions: {e}")
        return text


def data_preprocessing_of_the_review(text):
    try:
        # Expand contractions
        text = expand_contractions(text, contractions_dict)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        additional_stopwords = {"first", "time", "almost", "it", "in", "the", "to", "of", "for", "on", "a", "an"}
        stop_words = stop_words.union(additional_stopwords)
        critical_words = {"not", "no", "never"}
        tokens = [word for word in tokens if word not in stop_words or word in critical_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        processed_text = ' '.join(tokens)
        return processed_text
    except Exception as e:
        print(f"Error in preprocessing text: {e}")
        return text

if __name__ == '__main__':
    app.run(debug=True)
