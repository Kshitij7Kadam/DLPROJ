### YOUR CODE HERE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Network import DenseNet

class MyModel(torch.nn.Module):
    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.model = DenseNet(
            growth_rate=configs.get("growth_rate", 32),
            block_config=configs.get("block_config", (6, 12, 24, 16)),
            num_init_features=configs.get("num_init_features", 64),
            bn_size=configs.get("bn_size", 4),
            drop_rate=configs.get("drop_rate", 0),
            num_classes=configs.get("num_classes", 10)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=configs.get("lr", 0.001))

    def train(self, x_train, y_train, x_valid, y_valid, epochs=10, batch_size=64):
        # Creating training dataset and loader
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Ensuring x_valid and y_valid are tensors and not accidentally overwritten
        if isinstance(x_valid, torch.Tensor) and isinstance(y_valid, torch.Tensor):
            valid_dataset = TensorDataset(x_valid, y_valid)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        else:
            raise ValueError("x_valid or y_valid is not a tensor. Please check the data preparation process.")

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.3f}')
                    running_loss = 0.0

            if valid_loader:
                val_loss, val_acc = self.evaluate(valid_loader)
                print(f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_acc:.2f}%')

    def evaluate(self, data_loader):
        self.model.eval()
        total = 0
        correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def predict_prob(self, X):
        self.model.eval()
        X = torch.stack(X).to(self.device)
        outputs = self.model(X)
        return torch.softmax(outputs, dim=1)
### END CODE HERE

