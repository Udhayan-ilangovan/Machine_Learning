# Thompson sampling

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
u_dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

val = 10
total_reward_average_arr = [0]*val
for x in range(0,val):
    N = 10000
    d = 10
    number_of_rewards_1 = [0]*d
    number_of_rewards_0 = [0]*d
    add_selected = []
    reward_per_row = []
    rewarded_ad = []
    total_reward = 0
    ad_arr = []
    success_percentage = [0] * d
    upper_bond_arr=[]
    for i in range (0,N):
        ad = 0
        max_random = 0
        for j in range (0,d):
            random_beta = random.betavariate(number_of_rewards_1[j] + 1, number_of_rewards_0[j] + 1)
            if random_beta > max_random:
                max_random = random_beta
                c = i,j
                ad_arr.append(c)
                ad = j
        add_selected.append(ad)
        
        reward = u_dataset.values[i, ad]
        v = i,ad,reward
        reward_per_row.append(v)
        total_reward = total_reward + reward
        if reward == 1:
            number_of_rewards_1[ad] = number_of_rewards_1[ad]+ 1
        else:
            number_of_rewards_0[ad] = number_of_rewards_0[ad]+ 1
    
    
    
    success_percentage_TOTAL = 0
    for s in range (0,10):
        success_percentage[s] = round((number_of_rewards_1[s]/N)*100,2)
        success_percentage_TOTAL = success_percentage_TOTAL + success_percentage[s]
        total_reward_average_arr[x] = total_reward

total_reward_average = 0
for y in range(0,val):
    total_reward_average = total_reward_average + total_reward_average_arr[y] 
total_reward_average = total_reward_average/val

    
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

plt.figure("Histogram of ads selections")
plt.hist(add_selected)
plt.axes().yaxis.grid(True)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

