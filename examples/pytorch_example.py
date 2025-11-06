#!/usr/bin/env python3
"""
PyTorch Integration Example
============================

Train a simple neural network on DoOR data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from door_toolkit import DoOREncoder


class OdorDataset(Dataset):
    """Dataset wrapper for DoOR odorants."""
    
    def __init__(self, odor_names, labels, cache_path="door_cache"):
        self.encoder = DoOREncoder(cache_path, use_torch=True)
        self.odor_names = odor_names
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.odor_names)
    
    def __getitem__(self, idx):
        pn_activation = self.encoder.encode(self.odor_names[idx])
        return pn_activation, self.labels[idx]


class SimpleClassifier(nn.Module):
    """Simple feedforward classifier."""
    
    def __init__(self, n_receptors=78, n_hidden=256, n_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_receptors, n_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_hidden // 2, n_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def main():
    print("="*70)
    print("DoOR PyTorch Example - Odor Classification")
    print("="*70)
    
    # Check PyTorch availability
    if not torch.cuda.is_available():
        print("\nNote: CUDA not available, using CPU")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create synthetic dataset (replace with real labels)
    print("\n[1] Creating dataset...")
    encoder = DoOREncoder("door_cache", use_torch=True)
    
    # Get some odorants
    all_odorants = encoder.list_available_odorants()[:100]
    
    # Synthetic labels (replace with real task)
    # E.g., 0 = attractive, 1 = aversive
    import random
    labels = [random.randint(0, 1) for _ in all_odorants]
    
    # Train/test split
    split_idx = int(0.8 * len(all_odorants))
    train_odors, test_odors = all_odorants[:split_idx], all_odorants[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets
    train_dataset = OdorDataset(train_odors, train_labels)
    test_dataset = OdorDataset(test_odors, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"    Train: {len(train_dataset)} samples")
    print(f"    Test: {len(test_dataset)} samples")
    
    # Create model
    print("\n[2] Creating model...")
    model = SimpleClassifier(n_receptors=encoder.n_channels, n_classes=2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n[3] Training...")
    n_epochs = 10
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_y).sum().item()
        
        train_acc = train_correct / len(train_dataset)
        
        # Evaluate
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                test_correct += (outputs.argmax(1) == batch_y).sum().item()
        
        test_acc = test_correct / len(test_dataset)
        
        print(f"    Epoch {epoch+1}/{n_epochs}: "
              f"Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}")
    
    print("\n[4] Example predictions...")
    model.eval()
    with torch.no_grad():
        for odor in test_odors[:5]:
            pn = encoder.encode(odor).unsqueeze(0).to(device)
            logits = model(pn)
            pred = logits.argmax(1).item()
            conf = torch.softmax(logits, dim=1).max().item()
            print(f"    {odor[:30]:30s} -> Class {pred} (conf={conf:.3f})")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
