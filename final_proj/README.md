
# And they lived happily ever after?

Self-assigned project

Link to repository: https://github.com/ceciliestilling/CDS_LANG/edit/main/final_proj

## Project description

With this project, I wanted to do sentiment analysis on fairy tales to see how the polarity scores progressed throughout each individual fairy tale. It would be cool to use this tool to further investigate whether the polarity scores can reflect whether the fairy tale has a happy ending or not. However, there was no time to 
read all the fairy tales and assign endings to them, so this script only calculates the polarity score over time and plots them. 

## Methods
Each fairy tale was divided into chunks of sentences, and 
The chunks caused quite a bit of trouble, so I ended up using cumulative sum to create the chunks. 
Then the polarity scores were calculated using spacytextblob
The final scores were saved as a csv and plots for each of the 63 fairy tales were saved.

## Usage
Link to dataset: https://www.kaggle.com/code/kerneler/starter-grimms-fairy-tales-d3d7c7dd-b/data
The 'in' folder scructure should be: "in/grimms_fairytales.csv.csv".

To run this script through terminal, navigate to the folder outside the 'src' folder and run:

python3 src/cds_lang_proj.py

## Results
To find the output, navigate to the out folder. It should contain the following:

- a csv called 'final_scores.csv'
- 63 png files called 'polarityX.png' (X being the fairy tale number)
