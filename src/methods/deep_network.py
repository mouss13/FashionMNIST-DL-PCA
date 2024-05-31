import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_units=256, hidden_layers=3, activation=F.relu):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """

        super(MLP, self).__init__()
        self.activation = activation
        layers = [nn.Linear(input_size, hidden_units), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units, n_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        return self.model(x)

class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, conv_layers=[32, 64], kernel_size=3):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """

        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        in_channels = input_channels
        for out_channels in conv_layers:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            in_channels = out_channels

        # Calculate the size after conv and pooling layers
        # Assuming input image size is 28x28, after two poolings (2x2) it should be 7x7
        final_conv_output_size = (28 // (2 ** len(conv_layers))) * (28 // (2 ** len(conv_layers))) * conv_layers[-1]
        self.fc1 = nn.Linear(final_conv_output_size, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        preds = self.fc2(x)
        return preds

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        super(MyViT, self).__init__()
        # Placeholder for ViT model - must be replaced with actual implementation
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(chw[0] * chw[1] * chw[2], hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, out_d)
        )

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        preds = self.model(x)
        return preds

class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """
    def __init__(self, model, lr, epochs, batch_size, device=torch.device('cpu')):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """

        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader)

    def train_one_epoch(self, dataloader):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """

        self.model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            logits = self.model(inputs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """

        self.model.eval()
        pred_labels = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs[0].to(self.device)  
                logits = self.model(inputs)
                pred_labels.append(logits.argmax(dim=1))
        return torch.cat(pred_labels)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch   
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()