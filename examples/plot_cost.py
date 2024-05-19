import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

plant=sys.argv[1]

df = pd.read_csv(f'data/{plant}/J.csv', header=None, names=['cost'])
if plant=="sym":
   cost_values = df['cost'].apply(lambda x: float(x.strip('[]')))
else:
    cost_values = df['cost']
plt.figure(figsize=(20, 6))
sns.scatterplot(x=range(len(cost_values)), y=cost_values)
plt.title('Scatter Plot of Cost Values for last iteration')
plt.xlabel('N')
plt.ylabel('Cost')
plt.ylim(0,2)

plt.savefig(f'results/{plant}/J')

# for j in range(378):
#     try:
#         df = pd.read_csv(f'data/arm/cost{j}.csv', header=None, names=['cost'])
#         cost_values = df['cost'].apply(lambda x: float(x.strip('[]')))
#         plt.figure(figsize=(10, 6))
#         sns.scatterplot(x=range(len(cost_values)), y=cost_values)
#         plt.title('Scatter Plot of Cost Values')
#         plt.xlabel('time')
#         plt.ylabel('Cost')

#         plt.savefig(f'plots/cost{j}')
#     except Exception as e:
#         print(j, e)
   

    
