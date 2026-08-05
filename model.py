import torch
from torch import nn

class ReadmissionModel(nn.Module):
    """
    neural network model for predicting 
    diabetic patient readmission within 30 days

    """

    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
        )

    def forward(self, features):
        logits = self.network(features)

        return logits.squeeze(1)
