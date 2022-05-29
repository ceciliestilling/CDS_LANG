# system tools
import os
import sys
sys.path.append(os.path.join("utils"))

# data munging tools
import pandas as pd
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


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



# Vectorizing and Feature Extraction
#Create vectorizer object
vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams (1 word and 2 word units)
                             lowercase =  True,       
                             max_df = 0.95,           # remove very common words
                             min_df = 0.05,           # remove very rare words
                             max_features = 100)      # keep only top 500 features


# vectorizer turns all of our documents into a vector of numbers, instead of text
# first we fit to the training data
X_train_feats = vectorizer.fit_transform(X_train)

# then do it for our test data
X_test_feats = vectorizer.transform(X_test)

# get feature names
feature_names = vectorizer.get_feature_names()


# Classifying and predicting
classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)

# Make predictions using classifyer
y_pred = classifier.predict(X_test_feats)

# Inspecting the model to see which features are most informative when trying to predict a label
clf.show_features(vectorizer, y_train, classifier, n=20)

# Evaluate
# calculations to assess just how well our model performs
clf.plot_cm(y_test, y_pred, normalized=True)

# Make a classification report
cl_report = metrics.classification_report(y_test, y_pred)
#save
with open('out/cl_report_pt1.txt', 'w', encoding='UTF8') as f:
    f.write(cl_report)


