import json
import numpy as np
import nltk
import pickle
import random
from nltk.stem.snowball import FrenchStemmer
from tensorflow.keras.models import load_model

stemmer = FrenchStemmer()
model = None
intents = None
words = None
classes = None


def load_resources():
    global model, intents, words, classes
    model = load_model('chatbot_model.keras')
    with open('Data/intents.json', encoding='utf-8') as f:
        intents = json.load(f)
    with open('training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    words = data['words']
    classes = data['classes']


def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence, language='french')
    return [stemmer.stem(w.lower()) for w in tokens]


def bag_of_words(sentence):
    sentence_words = tokenize_and_stem(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    results = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]


def get_response(intents_list):
    if not intents_list:
        return "Désolé, je ne comprends pas votre demande. Pouvez-vous reformuler ?"
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Je n'ai pas de réponse enregistrée pour cela."
