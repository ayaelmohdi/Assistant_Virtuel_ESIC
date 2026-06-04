import json
import numpy as np
import nltk
import random
import pickle
from nltk.stem.snowball import FrenchStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

DATA_FILE = 'Data/intents.json'

with open(DATA_FILE, encoding='utf-8') as f:
    data = json.load(f)

print(f"Fichier {DATA_FILE} chargé.")

stemmer = FrenchStemmer()

def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence, language='french')
    return [stemmer.stem(w.lower()) for w in tokens]

words, classes, documents = [], [], []

for intent in data['intents']:
    tag = intent['tag']
    classes.append(tag)
    for pattern in intent['patterns']:
        stemmed = tokenize_and_stem(pattern)
        words.extend(stemmed)
        documents.append((stemmed, tag))

words = sorted(set(words))
classes = sorted(set(classes))

print(f"Documents : {len(documents)} | Classes : {len(classes)} | Vocabulaire : {len(words)}")

training_data = []
for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1
    training_data.append([bag, output_row])

random.shuffle(training_data)
training_data = np.array(training_data, dtype=object)
train_x = list(training_data[:, 0])
train_y = list(training_data[:, 1])

model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

print("Début de l'entraînement...")
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
print("Entraînement terminé.")

model.save('chatbot_model.keras')
pickle.dump({'words': words, 'classes': classes}, open('training_data.pkl', 'wb'))
print("Modèle et données sauvegardés.")
