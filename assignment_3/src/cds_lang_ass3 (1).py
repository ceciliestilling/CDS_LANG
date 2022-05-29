# System tools
import os
import sys

# Data analysis
import pandas as pd
from collections import Counter
from itertools import combinations 
from tqdm import tqdm

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")

# Network analysis tools
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (5,5)




try:
    input_file = sys.argv[1]
except:
    input_file = "in/1H4.csv"
    
    
edges = pd.read_csv(input_file, delimiter="\t")
    
# Create a graph object called G
G = nx.from_pandas_edgelist(edges, "Source", "Target", ["Weight"])

# Draw network
nx.draw_networkx(G, with_labels=True, node_size=20, font_size=10)

# save simple visualization
outpath_viz = os.path.join('out', 'viz',' network.png')
plt.savefig(outpath_viz, dpi=300, bbox_inches="tight")
    
# Centrality measures

# Eigenvector
ev = nx.eigenvector_centrality(G)
eigenvector_df = pd.DataFrame(ev.items())
eigenvector_df.sort_values(1, ascending=False)
eigenvector_df = eigenvector_df.rename(columns={eigenvector_df.columns[1]: 'EV'})

# Betweenness
bc = nx.betweenness_centrality(G)
betweenness_df = pd.DataFrame(bc.items()).sort_values(1, ascending=False)
betweenness_df = betweenness_df.rename(columns={betweenness_df.columns[1]: 'BC'})

# Merge data frames 
cent_meas_df = pd.merge(eigenvector_df, betweenness_df, on=[0])
cent_meas_df = cent_meas_df.rename(columns={cent_meas_df.columns[0]: 'Name'})

# Save as csv
cent_meas_df.to_csv(os.path.join("out", "cent_meas.csv"))
    
