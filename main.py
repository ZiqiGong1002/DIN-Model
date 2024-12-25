import pandas as pd
from dataset import RecommendationDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Load preprocessed DataFrame
data = pd.read_csv('Dataset/final_data.csv')

# Initialize the dataset
dataset = RecommendationDataset(data)

# Call the new coverage calculation method
coverage = dataset.__calculate_coverage__()
print("Coverage Information:")
print("Click Sequence Coverage:", coverage["click_coverage"])
print("Purchase Sequence Coverage:", coverage["purchase_coverage"])

# Test the dataset
print("Dataset size:", len(dataset))
sample = dataset[0]  # Get the first sample

# Print the sample's content
print("Discrete Features:", sample['discrete'])
print("Click Sequence:", sample['click_seq'])
print("Purchase Sequence:", sample['purchase_seq'])
print("Label:", sample['label'])

# Create DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Test DataLoader
for batch in dataloader:
    print("Batch - Discrete Features:", batch['discrete'].shape)
    print("Batch - Click Sequences:", batch['click_seq'].shape)
    print("Batch - Purchase Sequences:", batch['purchase_seq'].shape)
    print("Batch - Labels:", batch['label'].shape)
    break  # Print the first batch only
