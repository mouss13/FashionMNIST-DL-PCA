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

    def __init__(self, input_size, n_classes, nbLayer = 3 , activationFunction = F.relu):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        self.layers = nn.ModuleList()
        current_size = input_size

        # Add intermediate layers
        for _ in range(nbLayer - 1):
            self.layers.append(nn.Linear(current_size, 1024))
            current_size = 1024
        
        # Add the final layer
        self.layers.append(nn.Linear(current_size, n_classes))
        self.activationFunction = activationFunction
    

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        for layer in self.layers[:-1]:
            x = self.activationFunction(layer(x))
        return self.layers[-1](x)


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        self.model_CNN = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),  # ZAC 27.05 : corrected dimensions 
            nn.ReLU(),
            nn.Linear(128, n_classes),
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
        x = self.model_CNN(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        #preds: torch.Tensor = x
        #return preds
        return x


# Helper classes for the ViT
class PatchEmbedding(nn.Module):
    def __init__(self, chw, n_patches, hidden_d):
        super().__init__()
        self.ch, self.h, self.w = chw
        self.patch_size = self.h // int(n_patches**0.5)
        self.n_patches = (self.h // self.patch_size) * (self.w // self.patch_size)
        self.linear = nn.Linear(self.ch * self.patch_size * self.patch_size, hidden_d)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(B, C, self.n_patches, -1)
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        embeddings = self.linear(patches)
        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, n_patches, hidden_d):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, hidden_d))

    def forward(self, x):
        return x + self.pos_embedding


class TransformerBlock(nn.Module):
    def __init__(self, hidden_d, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_d, n_heads)
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, 4 * hidden_d),
            nn.GELU(),
            nn.Linear(4 * hidden_d, hidden_d),
        )
        self.norm2 = nn.LayerNorm(hidden_d)

    def forward(self, x):
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x



class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches, n_blocks, hidden_d, n_heads, out_d):
        """
        Initialize the network.
        
        """
        '''super().__init__()
        self.patch_embedding = PatchEmbedding(chw, n_patches, hidden_d)
        self.positional_encoding = PositionalEncoding(n_patches, hidden_d)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )
        self.fc = nn.Linear(hidden_d, out_d)'''

        super().__init__()
        self.patch_embedding = PatchEmbedding(chw, n_patches, hidden_d)
        self.n_patches = self.patch_embedding.n_patches
        self.positional_encoding = PositionalEncoding(self.n_patches, hidden_d)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )
        self.fc = nn.Linear(hidden_d, out_d)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)  # Global average pooling
        preds = self.fc(x)
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
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

        self.criterion = nn.CrossEntropyLoss()
        ### WRITE YOUR CODE HERE
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep=ep)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()  # set model to training mode
        for it, batch in enumerate(dataloader):
            # Fet the inputs and labels
            inputs, labels = batch

            # Run the forward pass
            logits = self.model.forward(inputs)

            # Compute the loss
            loss = self.criterion(logits, labels)

            # Compute the gradients
            loss.backward()

            # Update the weights
            self.optimizer.step()

            # Reset the gradients
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
        self.model.eval()  # set model to evaluation mode
        pred_labels = torch.tensor([]).long()
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                x = batch[0]
                pred = self.model(x)
                pred_labels = torch.cat((pred_labels, pred))
        pred_labels = torch.argmax(pred_labels, axis=1)

        return pred_labels
    
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