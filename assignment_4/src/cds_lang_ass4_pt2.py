# simple text processing tools
import re
import os
import tqdm
import unicodedata
import contractions
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
import sys
sys.path.append(os.path.join("utils"))
import utils.classifier_utils as clf


# data wranling
import pandas as pd
import numpy as np

# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.regularizers import L2

# scikit-learn
from sklearn.metrics import (confusion_matrix, 
                            classification_report)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# visualisations 
import matplotlib.pyplot as plt


# fix random seed for reproducibility
seed = 42
np.random.seed(seed)


# Helper functions for text processing - Ross
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
  
    return norm_docs

# read data
file = os.path.join("in", "VideoCommentsThreatCorpus.csv")
data = pd.read_csv(file)

# Check shape of data
shape = data.shape

# Check label count
label_count = data["label"].value_counts()

# Create balanced data
data_balanced = clf.balance(data, 1000)
data_balanced.shape

# Check label counts of balanced data
data_balanced["label"].value_counts()


# Let's now create new variables called X and y
X = data_balanced["text"]
y = data_balanced["label"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,               # texts for the model
                                                    y,               # classification labels
                                                    test_size=0.2,   # create an 80/20 split
                                                    random_state=42) # random state for reproducibility

# Clean data and normalize
X_train_norm = pre_process_corpus(X_train)
X_test_norm = pre_process_corpus(X_test)


#Preprocessing
# define out-of-vocabulary token
t = Tokenizer(oov_token = '<UNK>')

# fit the tokenizer on then documents
t.fit_on_texts(X_train_norm)

# set padding value
t.word_index["<PAD>"] = 0

#Tokenize sequences
X_train_seqs = t.texts_to_sequences(X_train_norm)
X_test_seqs = t.texts_to_sequences(X_test_norm)

print(f"Vocabulary size={len(t.word_index)}")
print(f"Number of Documents={t.document_count}")

train_lens = [len(s) for s in X_train_seqs]
test_lens = [len(s) for s in X_test_seqs]

fig, ax = plt.subplots(1,2, figsize=(12, 6))
h1 = ax[0].hist(train_lens)
h2 = ax[1].hist(test_lens)

# Sequence normalization
MAX_SEQ_LEN = 1000
# add padding to sequences
X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen=MAX_SEQ_LEN, padding="post")
X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen=MAX_SEQ_LEN, padding="post")
X_train_pad.shape, X_test_pad.shape


#Encoding labels
# create one-hot encodings - Do not use!
lb = LabelBinarizer()
y_train_lb = lb.fit_transform(y_train)
y_test_lb = lb.fit_transform(y_test)

# Create and compile model
# define paramaters for model
# overall vocublarly size
VOCAB_SIZE = len(t.word_index)
# number of dimensions for embeddings
EMBED_SIZE = 300
# number of epochs to train for
EPOCHS = 5
# batch size for training
BATCH_SIZE = 128

# create the model
model = Sequential()
# embedding layer
model.add(Embedding(VOCAB_SIZE, 
                    EMBED_SIZE, 
                    input_length=MAX_SEQ_LEN))

# first convolution layer and pooling
model.add(Conv1D(filters=128, 
                        kernel_size=4, 
                        padding='same',
                        activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# second convolution layer and pooling
model.add(Conv1D(filters=64, 
                        kernel_size=4, 
                        padding='same', 
                        activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=32, 
                        kernel_size=4, 
                        padding='same', 
                        activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# fully-connected classification layer
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
                        optimizer='adam', 
                        metrics=['accuracy'])
# print model summary
#model.summary()

# Train
model.fit(X_train_pad, y_train_lb,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        validation_split = 0.1,
        verbose = True)

# Evaluate
# Final evaluation of the model
scores = model.evaluate(X_test_pad, y_test_lb, verbose=1)
print(f"Accuracy: {scores[1]}")

# 0.5 decision boundary
predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
# confusion matrix and classification report

labels = [0, 1]
cl_report = classification_report(y_test, predictions)
print("\nClassification report:\n", cl_report)

# SAVE CLASSIFICATION REPORT
with open('out/cl_report_pt2.txt', 'w', encoding='UTF8') as f:
    f.write(cl_report)