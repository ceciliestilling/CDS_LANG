# Assignment 3 - Network Analysis

Cecilie Stilling Pedersen CDS_LANGUAGE_ANALYTICS

Link to repository: https://github.com/ceciliestilling/CDS_LANG/tree/main/assignment_3

## Assignment description

Your script should do the following:
- Load edgelist
- Perform network analysis using networkx
- Save a simple visualisation
- Save a CSV which shows the following for every node: name; degree; betweenness centrality; eigenvector_centrality

## Usage
The data "network_data" should be available to the examiner through our professor Ross.

The 'in' folder scructure should be: "in/network_data.csv".

To run this script through terminal, navigate to the folder outside the 'src' folder and run:

python3 src/cds_lang_ass3.py 'in/EDGELIST.csv'

### Example:
To run the script on edgelist 1H4.csv, navigate to the folder outside the 'src' folder and run:
python3 src/cds_lang_ass3.py 'in/1H4.csv'

## Results
To find the output, navigate to the out folder. It should contain the following:
- cent_meas.csv
- a folder called viz with _network.png in it

