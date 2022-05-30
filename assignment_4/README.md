# Assignment 4: Text Classification

Cecilie Stilling Pedersen
CDS_LANGUAGE_ANALYTICS

Link to repository: https://github.com/ceciliestilling/CDS_LANG/tree/main/assignment_4


## Assignment description
The assignment for this week builds on these concepts and techniques. We're going to be working with the data in the folder CDS-LANG/toxic and trying to see if we can predict whether or not a comment is a certain kind of toxic speech. You should write two scripts which do the following:

The first script should perform benchmark classification using standard machine learning approaches
- This means CountVectorizer() or TfidfVectorizer(), LogisticRegression classifier
- Save the results from the classification report to a text file
The second script should perform classification using the kind of deep learning methods we saw in class
- Keras Embedding layer, Convolutional Neural Network
- Save the classification report to a text file

## Usage
The data 'VideoCommentsThreatCorpus.csv' should be available to the examiner through our professor Ross, but can otherwise be found through this article: https://www.simula.no/sites/default/files/publications/files/cbmi2019_youtube_threat_corpus.pdf

The 'in' folder scructure should be: "in/VideoCommentsThreatCorpus.csv".

### Run script 1
To run this script through terminal, navigate to the folder outside the 'src' folder and run:

python3 src/cds_lang_ass4_pt1.py

### Run script 2

To run this script through terminal, navigate to the folder outside the 'src' folder and run:

python3 src/cds_lang_ass4_pt2.py

## Results

To find the output, navigate to the out folder. It should contain the following:
- cl_report_pt1.txt
- cl_report_pt2.txt





