# Assignment 2 - Sentiment and NER (Choice 2: FAKE VS REAL NEWS)

Cecilie Stilling Pedersen
CDS_LANGUAGE_ANALYTICS

Link to repository: https://github.com/ceciliestilling/CDS_LANG/tree/main/assignment_2 

## Assignment description

Using the corpus of Fake vs Real news, write some code which does the following

- Split the data into two datasets - one of Fake news and one of Real news
- For every headline
- Get the sentiment scores
- Find all mentions of geopolitical entites
- Save a CSV which shows the text ID, the sentiment scores, and column showing all GPEs in that text
- Find the 20 most common geopolitical entities mentioned across each dataset - plot the results as a bar charts



## Methods

This code takes the Fake vs Real news data set and splits it into two. Then for every headline of the two data sets, the sentiment scores are retrieved using spacytextblob, mentions of geopolitical entities are found and a CSV showing the text ID, the sentiment scores, and column showing all GPEs is saved. 
Finally, the 20 most common geopolitical entities mentioned across each data set is found and the results are plotted as bar charts.


## Usage
The data "fake_or_real_news" should be available to the examiner through our professor Ross.




The 'in' folder scructure should be: "in/fake_or_real_news.csv".

To run this script through terminal, navigate to the folder outside the 'src' folder and run:

python3 src/cds_lang_ass2.py


## Results 
To find the output, navigate to the out folder. It should contain the following:
- fake_output.csv
- gpe_count_fake.png
- gpe_count_real.png
- real_output.csv
