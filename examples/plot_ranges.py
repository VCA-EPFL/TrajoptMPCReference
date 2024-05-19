
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('gradient.csv', header=None)

# Initialize the figure and subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Iterate over the columns (elements)
for i in range(6):
    # Extract elements from the ith column, handling potential missing elements
    elements = []
    for index, row in df.iterrows():
        if len(row[0].strip('[]').split()) >= 6:
            elements.append(float(row[0].strip('[]').split()[i]))
        else:
            elements.append(None)
    
    # Determine the subplot position
    row = i // 3
    col = i % 3
    
    # Create the box plot in the corresponding subplot
    axs[row, col].boxplot(elements)
    axs[row, col].set_title(f'Element {i+1}')
    axs[row, col].set_xlabel('Rows')
    axs[row, col].set_ylabel('Value')

# Adjust layout
plt.tight_layout()
plt.savefig('gradient_ranges')
