import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")

# sentiment analysis VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# sentiment with spacyTextBlob
from spacytextblob.spacytextblob import SpacyTextBlob
nlp.add_pipe('spacytextblob')

# visualisations
import matplotlib.pyplot as plt



# load data
nRowsRead = 1000
data = pd.read_csv('..//final_proj/in/grimms_fairytales.csv', delimiter=',', nrows = nRowsRead)
data.dataframeName = 'grimms_fairytales.csv'
nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns')


n = 10     # number of chunks fairy tale is divided into
n_ft = 63  # number of fairy tales
final_score = []

for ft in range(0,n_ft):
    # Divide fairy tale into chunks
    sent_list = []
    chunks = [0]*n   # list of "0" of length n
    text = data["Text"][ft]
    doc = nlp(text)
    for number, sent in enumerate(doc.sents):
        # print(number, sent)
        length = len(list(doc.sents))
        #print(length)
        sent_list.append(sent)
        #print(len(sent_list))
    
    # Create the chunks using cumulative sum
    # Find the largest integer s.t. each chunk size are the same
    chunk_len = math.floor(length/100*n) 
    chunk_size = [chunk_len]*n          # Create n chunks of equal size
    rest = len(sent_list) - n*chunk_len # Find the remainder
    # Add the remaing to the chuck size 
    for r in range(0,rest):
        chunk_size[r]= chunk_size[r] + 1
    cs = [0]*11       # creating and calculating the cumulative sum
    for l in range(1,n):
        cs[l] = cs[l-1] + chunk_size[l-1]
    # Creating the n chunks of text of the specific chick size    
    for k in range(0,n):
        chunks[k] = sent_list[cs[k]:cs[k+1]] 

    # Convert to doc.doc type
    tmp = []
    for chunk in range(0,n):
        doc1 = nlp(str(chunks[chunk]))
        tmp.append(doc1)
    
    # calculate polarity score
    polarity = []
    for chunk in range(0,n):
        sentence = tmp[chunk]
        score = sentence._.blob.polarity
        polarity.append(score)
    
    final_score.append(polarity)
    
    
# make and save polarity plots
for i in range(0,63):
    plt.plot(final_score[i])
    plt.savefig(f'out/polarity{i}.png',dpi=100)
    #plt.show()
    plt.clf()
    
    
# create final_scores dataframe
final_scores_df = pd.DataFrame(final_score, columns = ['chunk0', 'chunk1', 'chunk2', 'chunk3', 'chunk4', 'chunk5', 'chunk6', 'chunk7', 'chunk8', 'chunk9'])
final_scores_df

# merge final_scores dataframe and original df
data_w_final_scores = data.join(final_scores_df)


# save as csv
data_w_final_scores.to_csv("out/final_scores.csv")