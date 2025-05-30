import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm
from pattern_database import PDBCollection
from blocksworld_domain import BlocksWorld15Domain
import random

class ImprovedNN(nn.Module):
    """Neural network matching paper's architecture."""
    
    def __init__(self, input_dim, hidden_dim=8, output_dim=2, dropout_prob=0.0):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ImprovedBayesianHeuristic:
    """Bayesian heuristic implementation."""
    
    def __init__(self, domain, input_dim, hidden_dim=8, l2_loss=True):
        self.domain = domain
        self.pdb_collection = PDBCollection(domain)
        self.solve_net = ImprovedNN(input_dim, hidden_dim, 2, dropout_prob=0.0)
        self.optimizer = optim.Adam(self.solve_net.parameters(), lr=0.001)
        self.epistemic_net = ImprovedNN(input_dim, hidden_dim, 1, dropout_prob=0.1)
        self.epistemic_optimizer = optim.Adam(self.epistemic_net.parameters(), lr=0.001)
        self.memory_buffer = []
        self.max_buffer_size = 25000
        self.feature_scales = None
        self.l2_loss = l2_loss
        
    def get_features(self, state):
        """Extract features matching paper's MultHeuristic15."""
        features = []
        pdb_values = self.pdb_collection.get_heuristics(state)
        features.extend(pdb_values)
        features.append(self.domain.hamming_distance(state))
        features.append(len(self.domain.state_to_stacks(state)))
        return np.array(features, dtype=np.float32)
    
    def normalize_features(self, features):
        """Normalize features using max values from PDB construction."""
        if self.feature_scales is None:
            self.feature_scales = np.array(self.pdb_collection.max_values, dtype=np.float32)
        normalized = features.copy()
        for i in range(len(features)):
            if i < len(self.feature_scales) and self.feature_scales[i] > 0:
                normalized[i] = features[i] / self.feature_scales[i]
            else:
                normalized[i] = features[i]
        return normalized
    
    def predict_with_uncertainty(self, X, num_samples=100, alpha=None):
        """Make predictions with uncertainty estimation."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_normalized = []
        for i in range(X.shape[0]):
            X_normalized.append(self.normalize_features(X[i]))
        X_tensor = torch.tensor(np.array(X_normalized), dtype=torch.float32)
        
        self.solve_net.eval()
        with torch.no_grad():
            main_output = self.solve_net(X_tensor)
            mean_pred = main_output[:, 0]
            log_var_pred = main_output[:, 1]
            aleatoric = torch.exp(log_var_pred) + 1e-6
        
        self.epistemic_net.train()
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.epistemic_net(X_tensor)
                predictions.append(output[:, 0])
        
        predictions = torch.stack(predictions)
        epistemic_mean = torch.mean(predictions, dim=0)
        epistemic = torch.var(predictions, dim=0, unbiased=False) + 1e-6
        total_uncertainty = epistemic + aleatoric
        
        if alpha is not None:
            z_score = norm.ppf(alpha)
            likely_admissible = mean_pred - z_score * torch.sqrt(total_uncertainty)
            return likely_admissible.numpy(), mean_pred.numpy(), epistemic.numpy(), aleatoric.numpy()
        
        return mean_pred.numpy(), epistemic.numpy(), aleatoric.numpy(), total_uncertainty.numpy()
    
    def train_networks(self, X, y, epochs=1000):
        """Train the networks with early stopping."""
        X = np.array(X)
        y = np.array(y)
        X_normalized = []
        for i in range(X.shape[0]):
            X_normalized.append(self.normalize_features(X[i]))
        X_tensor = torch.tensor(np.array(X_normalized), dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        self.solve_net.train()
        best_loss = float('inf')
        patience = 50
        patience_counter = 0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.solve_net(X_tensor)
            mean_pred = output[:, 0]
            log_var_pred = output[:, 1]
            precision = torch.exp(-log_var_pred)
            if self.l2_loss:
                loss = torch.mean(0.5 * precision * (y_tensor - mean_pred) ** 2 + 0.5 * log_var_pred)
            else:
                loss = torch.mean(precision * torch.abs(y_tensor - mean_pred) + 0.5 * log_var_pred)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
            if epoch % 200 == 0:
                print(f'  Main network epoch {epoch}, loss: {loss.item():.4f}')
        
        self.epistemic_net.train()
        for epoch in range(200):
            self.epistemic_optimizer.zero_grad()
            output = self.epistemic_net(X_tensor)
            mean_pred = output[:, 0]
            loss = nn.MSELoss()(mean_pred, y_tensor)
            loss.backward()
            self.epistemic_optimizer.step()
    
    def add_to_memory_buffer(self, X, y):
        """Add states and their costs with better memory management."""
        if not isinstance(X, list):
            X = [X]
        if not isinstance(y, list):
            y = [y]
        for i in range(len(X)):
            self.memory_buffer.append((X[i], y[i]))
        if len(self.memory_buffer) % 1000 == 0:
            seen = set()
            unique_buffer = []
            for item in self.memory_buffer:
                item_key = (tuple(item[0]), item[1])
                if item_key not in seen:
                    seen.add(item_key)
                    unique_buffer.append(item)
            self.memory_buffer = unique_buffer
        if len(self.memory_buffer) > self.max_buffer_size:
            recent = self.memory_buffer[-self.max_buffer_size//2:]
            old = self.memory_buffer[:-self.max_buffer_size//2]
            if old:
                sampled_old = random.sample(old, min(len(old), self.max_buffer_size//2))
                self.memory_buffer = recent + sampled_old
            else:
                self.memory_buffer = recent