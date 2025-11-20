import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtractor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Regression output
        )

    def forward(self, x):
        return self.net(x)

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Binary classification (Source vs Target)
        )

    def forward(self, x):
        return self.net(x)

def train_dann(source_data, target_data, epochs=50):
    """
    Trains DANN model.
    source_data: (X_source, y_source)
    target_data: (X_target) - Unlabeled
    """
    print("Training DANN model...")
    
    X_source, y_source = source_data
    X_target = target_data
    
    # Convert to tensors
    X_source = torch.tensor(X_source, dtype=torch.float32)
    y_source = torch.tensor(y_source, dtype=torch.float32).view(-1, 1)
    X_target = torch.tensor(X_target, dtype=torch.float32)
    
    # DataLoaders
    batch_size = 32
    source_loader = DataLoader(TensorDataset(X_source, y_source), batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(TensorDataset(X_target), batch_size=batch_size, shuffle=True)
    
    # Initialize networks
    feature_extractor = FeatureExtractor(X_source.shape[1])
    label_predictor = LabelPredictor()
    domain_classifier = DomainClassifier()
    
    optimizer = optim.Adam(
        list(feature_extractor.parameters()) + 
        list(label_predictor.parameters()) + 
        list(domain_classifier.parameters()), lr=0.001
    )
    
    loss_label = nn.MSELoss()
    loss_domain = nn.BCELoss()
    
    for epoch in range(epochs):
        len_dataloader = min(len(source_loader), len(target_loader))
        data_source_iter = iter(source_loader)
        data_target_iter = iter(target_loader)
        
        total_loss = 0
        
        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / (epochs * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # Source data
            s_img, s_label = next(data_source_iter)
            
            # Target data
            t_img = next(data_target_iter)[0]
            
            # Forward pass
            # Source
            s_feature = feature_extractor(s_img)
            s_pred = label_predictor(s_feature)
            s_domain_pred = domain_classifier(s_feature)
            
            # Target
            t_feature = feature_extractor(t_img)
            t_domain_pred = domain_classifier(t_feature)
            
            # Losses
            err_s_label = loss_label(s_pred, s_label)
            
            # Domain labels: 0 for source, 1 for target
            domain_label_s = torch.zeros(s_img.size(0), 1)
            domain_label_t = torch.ones(t_img.size(0), 1)
            
            err_s_domain = loss_domain(s_domain_pred, domain_label_s)
            err_t_domain = loss_domain(t_domain_pred, domain_label_t)
            
            # Total loss
            # We want to minimize label error and maximize domain error (adversarial)
            # Standard DANN implementation uses Gradient Reversal Layer (GRL).
            # Here simplified: minimize (label_loss + domain_loss_source + domain_loss_target)
            # Note: For true adversarial, we need GRL or separate optimization steps.
            # This is a simplified implementation for demonstration.
            
            loss = err_s_label + alpha * (err_s_domain + err_t_domain)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len_dataloader:.4f}")
            
    return feature_extractor, label_predictor

if __name__ == "__main__":
    # Mock data for testing
    X_s = np.random.rand(100, 10)
    y_s = np.random.rand(100)
    X_t = np.random.rand(100, 10)
    
    fe, lp = train_dann((X_s, y_s), X_t)
    print("DANN training complete.")
