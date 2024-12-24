import torch
from torch.utils.data import Dataset

class RecommendationDataset(Dataset):
    def __init__(self, dataframe):
        """
        Initialize the custom PyTorch Dataset.
        :param dataframe: Preprocessed pandas DataFrame.
        """
        self.data = dataframe

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single sample by index.
        """
        row = self.data.iloc[idx]

        # Extract discrete features
        discrete_features = torch.tensor([
            row['user_id'],
            row['age_range'],
            row['gender'],
            row['cat_id'],
            row['merchant_id'],
            row['brand_id'],
            row['item_id']
        ], dtype=torch.long)

        # Extract click sequence features
        click_sequence = torch.tensor(
            [row[f'click_seq_{i}'] for i in range(1, 11)],
            dtype=torch.long
        )

        # Extract purchase sequence features
        purchase_sequence = torch.tensor(
            [row[f'purchase_seq_{i}'] for i in range(1, 11)],
            dtype=torch.long
        )

        # Extract label
        label = torch.tensor(row['label'], dtype=torch.float)

        # Return a dictionary
        return {
            'discrete': discrete_features,       # Discrete features, shape (7,)
            'click_seq': click_sequence,        # Click sequence, shape (10,)
            'purchase_seq': purchase_sequence,  # Purchase sequence, shape (10,)
            'label': label                      # Label, scalar
        }
