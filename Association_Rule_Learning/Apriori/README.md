# Apriori
* The Apriori algorithm works on datasets like transactional records or data including certain quantities of fields or items.
* It identifies the significant frequent individual item in the dataset.
* It extends the item sets as long as these itemsets appear frequently in the dataset.
* It is one of the association rule learning methods.
    * Association rule learning is a rule-based machine learning method for identifying unique relations between variables in a massive dataset.
* It uses a "bottom-up approach" to incrementally contrast hidden records.
* Example
    * Customer purchase dataset in a supermarket 
    * Input data : product purchased by customer (each row is one transaction)

## Apriori 
import pandas as pd # used for importing dataset and manage dataset

### Importing the dataset
ap_dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

X = ap_dataset.iloc[:,:].values

trancations = []

for i in range (0,7501):

        trancations.append([str(X[i,j]) for j in range (0,20)])
        
### Training the Aprori model
from apyori import apriori

min_sup_value =3

days = 7

total_order =7501

min_support_i = round(3*7/7501,3)

a_rules = apriori(trancations,min_support = min_support_i ,min_confidence = 0.2 ,min_lift = 3 ,min_length =2)

### Generating output
results = list(a_rules)

results_list = []

for i in range(0, len(results)): 

    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))        
            
### Exporting the rules to csv  

results_list_csv = pd.DataFrame(results_list)

results_list_csv.to_csv('results_list_csv.csv', index=False, header=False)


### Apriori sample dataset

<img width="1420" alt="apriori_dataset" src="https://user-images.githubusercontent.com/32480274/50400620-479e1580-0788-11e9-9511-49ebb906c513.png">

￼

### Apriori output (generated rules) in a CSV 



<img width="1435" alt="apriori_output_rule" src="https://user-images.githubusercontent.com/32480274/50400623-4bca3300-0788-11e9-9e2e-064fc2985cb6.png">

￼
