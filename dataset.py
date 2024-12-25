import torch
from torch.utils.data import Dataset

class RecommendationDataset(Dataset):
    def __init__(self, dataframe, sequence_length=10, padding_value=0):
        """
        Initialize the custom PyTorch Dataset.
        :param dataframe: Preprocessed pandas DataFrame.
        :param sequence_length: Length of the behavior sequences to consider.
        :param padding_value: The value used for padding in sequences.
        """
        self.data = dataframe
        self.sequence_length = sequence_length
        self.padding_value = padding_value

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __calculate_coverage__(self):
        """
        Calculate the coverage of valid user sequences.
        Coverage is defined as the ratio of non-padded elements in click/purchase sequences
        over the total sequence length for all users.
        """
        total_click_elements = 0
        valid_click_elements = 0
        total_purchase_elements = 0
        valid_purchase_elements = 0

        for _, row in self.data.iterrows():
            # Calculate click sequence coverage
            click_seq = [row[f'click_seq_{i}'] for i in range(1, self.sequence_length + 1)]
            total_click_elements += len(click_seq)
            valid_click_elements += sum(1 for x in click_seq if x != self.padding_value)

            # Calculate purchase sequence coverage
            purchase_seq = [row[f'purchase_seq_{i}'] for i in range(1, self.sequence_length + 1)]
            total_purchase_elements += len(purchase_seq)
            valid_purchase_elements += sum(1 for x in purchase_seq if x != self.padding_value)


        click_coverage = valid_click_elements / total_click_elements if total_click_elements > 0 else 0
        purchase_coverage = valid_purchase_elements / total_purchase_elements if total_purchase_elements > 0 else 0

        return {
            "click_coverage": click_coverage,
            "purchase_coverage": purchase_coverage
        }

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
            [row[f'click_seq_{i}'] for i in range(1, self.sequence_length + 1)],
            dtype=torch.long
        )

        # Extract purchase sequence features
        purchase_sequence = torch.tensor(
            [row[f'purchase_seq_{i}'] for i in range(1, self.sequence_length + 1)],
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
