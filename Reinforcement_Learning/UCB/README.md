# upper confident bound
* The upper confident bound algorithm is used in reinforcement learning. 
* The purpose of the algorithm is to maximize the total reward for the given set of rule or actions.
* Formulas
    * To calculate the average reward of advertisement

<img width="202" alt="ucb_average_reward_add" src="https://user-images.githubusercontent.com/32480274/50403953-2dbdfc00-07a3-11e9-8350-fc5618321f2e.png">
￼
<img width="309" alt="ucb_condfident_leve" src="https://user-images.githubusercontent.com/32480274/50403955-31518300-07a3-11e9-8e0a-2be5933e2422.png">
￼
    * To calculate the confidence level of advertisement

    * To select the advertisement which has maximum rewards
<img width="269" alt="ucb_maximum" src="https://user-images.githubusercontent.com/32480274/50403959-357da080-07a3-11e9-8a7b-90a4087aac67.png">
￼
* In our example, it will show an advertisement based on its rewards
    * In the start, it will show all advertisement to the viewer.
    * If the viewer clicks the advertisement, Then it is a success so that particular add get a reward.
    * If the viewer doesn't click the advertisement, Then that particular add doesn't get any reward.
    * So, it will calculate the confident level by comparing the number of times the advertisement is shown and that particular advertisement's reward the selection of add keep changing periodically. 
## UCB
import numpy as np # used for mathematical operations

import matplotlib.pyplot as plt # used for plot graphically

import pandas as pd # used for importing dataset and manage dataset

import math # used for mathematical operations like log operations

### Importing the dataset

u_dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

### Implementing UCB algorithm
N = 10000

d = 10

number_of_selections = [0] * d

sum_of_reward = [0] * d

add_selected = []

reward_per_row = []

rewarded_ad = []

total_reward = 0

ad_arr = []

number_of_selections_total = 0

success_percentage = [0] * d

upper_bond_arr=[]

for i in range (0,N):

    ad = 0
    max_upper_bound = 0

    for j in range (0,d):

        if (number_of_selections[j] > 0):
            average_reward = sum_of_reward[j]/number_of_selections[j]
            delta_i = math.sqrt((3/2) * (math.log(i + 1)/number_of_selections[j]))
            upper_bound = average_reward + delta_i
            up = j ,i , upper_bound
            upper_bond_arr.append(up)

        else:
             upper_bound = 1e400
             up = j ,i , upper_bound
             upper_bond_arr.append(up)

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            c = i,j
            ad_arr.append(c)
            ad = j

    add_selected.append(ad)

    number_of_selections[ad] = number_of_selections[ad]+1
    
    reward = u_dataset.values[i, ad]

    v = i,ad,reward

    reward_per_row.append(v)

    sum_of_reward[ad] = sum_of_reward[ad] + reward

    total_reward = total_reward + reward

for t in range (0,10):    

    number_of_selections_total = number_of_selections_total + number_of_selections[t]

for s in range (0,10):

    success_percentage[s] = round((sum_of_reward[s]/number_of_selections[s])*100,2)

### Visualising results in percentage

plt.figure("Histogram of ads selections in percentage")

x = []

for a in range (1,11):
    x.append(a)

plt.scatter(x,success_percentage)

plt.bar(x,success_percentage)

plt.xticks(np.arange(11))

plt.yticks(np.arange(28))

plt.axes().yaxis.grid(True)

plt.xlabel('Ads')

plt.ylabel('Percentage of selection')

plt.show()

### Visualising results in real numbers

plt.figure("Histogram of ads selections")

plt.hist(add_selected)

plt.axes().yaxis.grid(True)

plt.title('Histogram of ads selections')

plt.xlabel('Ads')

plt.ylabel('Number of times each ad was selected')

plt.show()

### Visualising
Number of time add selected out of 10000

<img width="1347" alt="ucb_add_selected_rn" src="https://user-images.githubusercontent.com/32480274/50403964-3f070880-07a3-11e9-8e62-a4bd387fd30d.png">
￼
### Visualising
Success percentage of add selected

<img width="1362" alt="ucb_add_selected_pr" src="https://user-images.githubusercontent.com/32480274/50403965-429a8f80-07a3-11e9-86b6-87232031092f.png">
