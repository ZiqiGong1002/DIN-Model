import torch
import torch.nn as nn
import torch.nn.functional as F

class DIN(nn.Module):
    def __init__(self, user_feature_sizes, item_feature_sizes, embedding_dim):
        """
        Initialize the Deep Interest Network (DIN).

        :param user_feature_sizes: List of vocab sizes for user features.
        :param item_feature_sizes: List of vocab sizes for item features.
        :param embedding_dim: Dimension of embedding vectors.
        """
        super(DIN, self).__init__()

        # Embedding layers for user features
        self.user_embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim) for size in user_feature_sizes
        ])

        # Embedding layers for item features
        self.item_embeddings = nn.ModuleList([
            nn.Embedding(size, embedding_dim) for size in item_feature_sizes
        ])

        # Attention mechanism to calculate relevance between user behavior and target item
        self.attention_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Final fully connected layers for prediction
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * (len(user_feature_sizes) + 2), 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user_features, target_item_features, behavior_sequence):
        """
        Forward pass for DIN.

        :param user_features: Tensor of shape (batch_size, num_user_features).
        :param target_item_features: Tensor of shape (batch_size, num_item_features).
        :param behavior_sequence: Tensor of shape (batch_size, seq_len, num_item_features).
        :return: Predicted scores, shape (batch_size, 1).
        """
        batch_size, seq_len, _ = behavior_sequence.size()

        # Embedding for user features
        user_embeds = torch.cat([
            emb(user_features[:, i]) for i, emb in enumerate(self.user_embeddings)
        ], dim=-1)  # Shape: (batch_size, num_user_features * embedding_dim)

        # Embedding for target item
        target_item_embed = torch.cat([
            emb(target_item_features[:, i]) for i, emb in enumerate(self.item_embeddings)
        ], dim=-1)  # Shape: (batch_size, embedding_dim)

        # Embedding for behavior sequence
        behavior_embeds = torch.cat([
            emb(behavior_sequence[:, :, i]) for i, emb in enumerate(self.item_embeddings)
        ], dim=-1)  # Shape: (batch_size, seq_len, embedding_dim)

        # Attention mechanism
        target_expanded = target_item_embed.unsqueeze(1).expand_as(behavior_embeds)  # Shape: (batch_size, seq_len, embedding_dim)
        attention_input = torch.cat([
            behavior_embeds, target_expanded, behavior_embeds - target_expanded, behavior_embeds * target_expanded
        ], dim=-1)  # Shape: (batch_size, seq_len, embedding_dim * 4)
        attention_scores = self.attention_mlp(attention_input).squeeze(-1)  # Shape: (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: (batch_size, seq_len)

        # Weighted sum of behavior embeddings
        user_interest = torch.sum(behavior_embeds * attention_weights.unsqueeze(-1), dim=1)  # Shape: (batch_size, embedding_dim)

        # Combine user features, user interest, and target item features
        combined_features = torch.cat([user_embeds, user_interest, target_item_embed], dim=-1)  # Shape: (batch_size, embedding_dim * (num_user_features + 2))

        # Final prediction
        output = self.fc(combined_features)  # Shape: (batch_size, 1)
        return output
