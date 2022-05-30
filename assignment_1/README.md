
# Assignment 1 - Collocation Tool

Cecilie Stilling Pedersen
CDS_LANGUAGE_ANALYTICS

Link to repository: https://github.com/ceciliestilling/CDS_LANG/tree/main/assignment_1

## Assignment description

* Take a user-defined search term and a user-defined window size.
* Take one specific text which the user can define.
* Find all the context words which appear ± the window size from the search term in that text.
* Calculate the mutual information score for each context word.
* Save the results as a CSV file with (at least) the following columns: the collocate term; how often it appears as a collocate; how often it appears in the text; the mutual information score.

## Methods
perform collocational analysis using simple string processing and NLP tools. The strength of the association between collocates is calculated using the formula described below.

### MI-score calculation formula:

MI = log ( (AB * sizeCorpus) / (A * B * span) ) / log (2)
 
where 
- A = frequency of node word:
- B = frequency of collocate:
- AB = frequency of collocate near the node word
- sizeCorpus= size of corpus
- span = span of words
- log (2) is the log10 of the number 2


## Usage
Link to the data can be found here: https://github.com/computationalstylistics/100_english_novels


To run this script in the terminal, navigate to the folder outside the 'src' folder and run:
python3 src/assignment1.py FILEPATH KEYWORD WINDOWSIZE
 
### EXAMPLE:
To run the script on the text Haggard_Mines_1885.txt with the keyword boy and the window size 3, navigate to the folder outside the 'src' folder and run:
  python3 src/cds_lang_ass1.py 'in/Haggard_Mines_1885.txt' boy 3
