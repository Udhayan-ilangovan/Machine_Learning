# Apriori 
import pandas as pd

# Importing the dataset
ap_dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
X = ap_dataset.iloc[:,:].values
trancations = []
for i in range (0,7501):
        trancations.append([str(X[i,j]) for j in range (0,20)])
        
# Training the Aprori model
from apyori import apriori
min_sup_value =3
days = 7
total_order =7501
min_support_i = round(3*7/7501,3)
a_rules = apriori(trancations,min_support = min_support_i ,min_confidence = 0.2 ,min_lift = 3 ,min_length =2)

# Generating output
results = list(a_rules)
results_list = []
for i in range(0, len(results)): 
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))        
            
# Exporting the rules to csv  
results_list_csv = pd.DataFrame(results_list)
results_list_csv.to_csv('results_list_csv.csv', index=False, header=False)