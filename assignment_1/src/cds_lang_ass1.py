# Import libraries
import spacy
import os
import sys
nlp = spacy.load("en_core_web_sm")
import string
import math
import csv


try:
    filename = sys.argv[1]
except:
    filename = "in/Stevenson_Catriona_1893.txt"


# Load one text from the data folder
#filename = os.path.join("in", "Stevenson_Catriona_1893.txt")
with open(filename, "r") as f:
    text = f.read() 
    
# Define search word, pos (word class) and window size:
try:
    keyword = sys.argv[2]
except: 
    keyword = "girl"
#pos = "NOUN"
try:
    window_size = sys.argv[3]
except:
    window_size = 2

# Normalization of text string:
# Make everything lower case
text_clean = text.lower()

# Remove all new lines
text_clean = text_clean.rstrip("\n") 

# Remove all punctuations
text_clean = text_clean.translate(str.maketrans('', '', string.punctuation)) 


# create doc
doc = nlp(text_clean)


# make function to remove keyword
def remove_i_element_from_span(span, index):
    nlp_list = list(span)
    del nlp_list[index]
    return nlp(" ".join([e.text for e in nlp_list]))

collocates = [] 

ws = int(window_size)

# Finding collocates
for token in doc:
    if token.text == keyword:
        before = token.i - ws     
        after = token.i + ws + 1 
        span = doc[before:after]
        #span_clean = remove_i_element_from_span(span,ws)
        collocates.append(span)
    else:
        pass
    
# function to calculate keyword frequency
def keyword_freq(token, text):
    freq = 0
    for word in text:
        if word.text == token.text:
            freq += 1
    return freq

# function to calculate collocate frequency
def collocate_freq(token, collocates):
    freq = 0
    for tok in collocates:
        span = tok
        for word in span:
            if token.text == word.text:
                freq += 1
    return freq

# function to calculate MI score
def MIS(A, B, AB, span, sizeCorpus):
    score = math.log((AB * sizeCorpus) / (A * B * span_size)) / math.log(2)
    return score


sizeCorpus = len(doc)
span_size = 2 * ws


output = []

for tok in collocates: 
    kw = tok[ws]
    A = keyword_freq(kw, doc)
    span_clean = remove_i_element_from_span(tok,ws)
    for word in span_clean:
        B = keyword_freq(word, doc)
        AB = collocate_freq(word, collocates)
        if AB != 0:
            # Calculate the Mutual Information score:
            MI_S = MIS(A, B, AB, span_size, sizeCorpus)
        else:
            pass

        #Append [collocate], [freq], [doc_freq], [mut_inf_sc]
        output.append([word, AB, B, MI_S])
        
        
output  
    
outfile = f"out/{keyword}_collocates.csv"

header = ["collocate", "frequency", "doc,freq", "MI score"] 

with open(outfile, 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

   # Write 
    for out in output:
        writer.writerow(out)