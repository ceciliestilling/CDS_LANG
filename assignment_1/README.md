
# Assignment 1 - Collocation Tool

* Take a user-defined search term and a user-defined window size.
* Take one specific text which the user can define.
* Find all the context words which appear ± the window size from the search term in that text.
* Calculate the mutual information score for each context word.
* Save the results as a CSV file with (at least) the following columns: the collocate term; how often it appears as a collocate; how often it appears in the text; the mutual information score.


EXAMPLE:
python3 src/cds_lang_ass1.py 'in/Haggard_Mines_1885.txt' boy 3
