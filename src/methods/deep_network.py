import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self, input_size, n_classes, nbLayer=3, activationFunction=F.relu):
        super().__init__()
        self.layers = nn.ModuleList()
        current_size = input_size
        for _ in range(nbLayer - 1):
            self.layers.append(nn.Linear(current_size, 1024))
            current_size = 1024
        self.layers.append(nn.Linear(current_size, n_classes))
        self.activationFunction = activationFunction

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activationFunction(layer(x))
        return self.layers[-1](x)

class CNN(nn.Module):
    def __init__(self, input_channels, n_classes):
        super().__init__()
        self.model_CNN = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifierz = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.model_CNN(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Trainer(object):
    def __init__(self, model, lr, epochs, batch_size, device):
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_all(self, dataloader):
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep=ep)

    def train_one_epoch(self, dataloader, ep):
        self.model.train()
        for it, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model.forward(inputs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

    def predict_torch(self, dataloader):
        self.model.eval()
        pred_labels = torch.tensor([]).long().to(self.device)
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                x = batch[0].to(self.device)
                pred = self.model(x)
                pred_labels = torch.cat((pred_labels, pred))
        pred_labels = torch.argmax(pred_labels, axis=1)
        return pred_labels

    def fit(self, training_data, training_labels):
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)
        return self.predict(training_data)

    def predict(self, test_data):
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        pred_labels = self.predict_torch(test_dataloader)
        return pred_labels.cpu().numpy()
