import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2
import torch.optim as optim

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
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
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = self.fc4(x)

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes,image_size=(28,28)):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN,self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after convolutional and pooling layers
        self.feature_map_size = self._get_feature_map_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * self.feature_map_size[0] * self.feature_map_size[1], 512)
        self.fc2 = nn.Linear(512, n_classes)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def _get_feature_map_size(self):
        # Create a dummy tensor with the input shape to calculate the size of the feature map
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, *self.image_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        return x.shape[2], x.shape[3]
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """      
        # Reshape the input tensor to (N, input_channels, H, W)
        x = x.view(-1, self.input_channels, *self.image_size)

        # Apply first convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply first fully connected layer with dropout and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Apply second fully connected layer
        preds = self.fc2(x)
        
        return preds

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        # assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T) / np.sqrt(self.d)                
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, n_classes=10):
        """
        Initialize the network.
        """        
        super(MyViT, self).__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, n_classes),
            nn.Softmax(dim=-1)
        )
    
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w # We assume square image.
        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches
    
    def get_positional_embeddings(self, sequence_length, d):
        """
        Get the positional embeddings for a given sequence length and hidden size.
        """
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = np.sin(i / 10000 ** (j / d))
                else:
                    result[i, j] = np.cos(i / 10000 ** (j / d))
        return result
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = x.view(-1, *self.chw)
        n, c, h, w = x.shape
        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)
        
        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) ### WRITE YOUR CODE HERE

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        out = tokens + self.positional_embeddings.unsqueeze(0).repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out


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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader,ep)


    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch: {ep} [{batch_idx * len(data)}/{len(dataloader.dataset)}'
                      f' ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
            # if batch_idx % 20 == 0:
            #     loss_dict.append(loss.item())
            # if (ep+1) % 5 == 0:
            #     torch.save(self.model.state_dict(), f'checkpoint/model_ep{ep}.pth')


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
            for data in dataloader:
                outputs = self.model(data[0])
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted)
        pred_labels = torch.cat(pred_labels)
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
    

def split_train_val(xtrain, ytrain, val_ratio=0.1):
    np.random.seed(42)
    N = xtrain.shape[0]
    Nval = int(N*val_ratio)
    indices = np.arange(xtrain.shape[0])
    np.random.shuffle(indices)

    xtrain = xtrain[indices]
    ytrain = ytrain[indices]
    xval = xtrain[:Nval]
    yval = ytrain[:Nval]
    xtrain = xtrain[Nval:]
    ytrain = ytrain[Nval:]
    
    return xtrain, ytrain, xval, yvalimport numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2
import torch.optim as optim

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
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
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = self.fc4(x)

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes,image_size=(28,28)):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN,self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after convolutional and pooling layers
        self.feature_map_size = self._get_feature_map_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * self.feature_map_size[0] * self.feature_map_size[1], 512)
        self.fc2 = nn.Linear(512, n_classes)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def _get_feature_map_size(self):
        # Create a dummy tensor with the input shape to calculate the size of the feature map
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, *self.image_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        return x.shape[2], x.shape[3]
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """      
        # Reshape the input tensor to (N, input_channels, H, W)
        x = x.view(-1, self.input_channels, *self.image_size)

        # Apply first convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply first fully connected layer with dropout and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Apply second fully connected layer
        preds = self.fc2(x)
        
        return preds

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        # assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T) / np.sqrt(self.d)                
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, n_classes=10):
        """
        Initialize the network.
        """        
        super(MyViT, self).__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, n_classes),
            nn.Softmax(dim=-1)
        )
    
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w # We assume square image.
        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches
    
    def get_positional_embeddings(self, sequence_length, d):
        """
        Get the positional embeddings for a given sequence length and hidden size.
        """
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = np.sin(i / 10000 ** (j / d))
                else:
                    result[i, j] = np.cos(i / 10000 ** (j / d))
        return result
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = x.view(-1, *self.chw)
        n, c, h, w = x.shape
        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)
        
        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) ### WRITE YOUR CODE HERE

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        out = tokens + self.positional_embeddings.unsqueeze(0).repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out


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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader,ep)


    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch: {ep} [{batch_idx * len(data)}/{len(dataloader.dataset)}'
                      f' ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
            # if batch_idx % 20 == 0:
            #     loss_dict.append(loss.item())
            # if (ep+1) % 5 == 0:
            #     torch.save(self.model.state_dict(), f'checkpoint/model_ep{ep}.pth')


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
            for data in dataloader:
                outputs = self.model(data[0])
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted)
        pred_labels = torch.cat(pred_labels)
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
    

def split_train_val(xtrain, ytrain, val_ratio=0.1):
    np.random.seed(42)
    N = xtrain.shape[0]
    Nval = int(N*val_ratio)
    indices = np.arange(xtrain.shape[0])
    np.random.shuffle(indices)

    xtrain = xtrain[indices]
    ytrain = ytrain[indices]
    xval = xtrain[:Nval]
    yval = ytrain[:Nval]
    xtrain = xtrain[Nval:]
    ytrain = ytrain[Nval:]
    
    return xtrain, ytrain, xval, yvalimport numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2
import torch.optim as optim

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
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
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = self.fc4(x)

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes,image_size=(28,28)):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN,self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after convolutional and pooling layers
        self.feature_map_size = self._get_feature_map_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * self.feature_map_size[0] * self.feature_map_size[1], 512)
        self.fc2 = nn.Linear(512, n_classes)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def _get_feature_map_size(self):
        # Create a dummy tensor with the input shape to calculate the size of the feature map
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, *self.image_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        return x.shape[2], x.shape[3]
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """      
        # Reshape the input tensor to (N, input_channels, H, W)
        x = x.view(-1, self.input_channels, *self.image_size)

        # Apply first convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply first fully connected layer with dropout and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Apply second fully connected layer
        preds = self.fc2(x)
        
        return preds

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        # assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T) / np.sqrt(self.d)                
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, n_classes=10):
        """
        Initialize the network.
        """        
        super(MyViT, self).__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, n_classes),
            nn.Softmax(dim=-1)
        )
    
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w # We assume square image.
        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches
    
    def get_positional_embeddings(self, sequence_length, d):
        """
        Get the positional embeddings for a given sequence length and hidden size.
        """
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = np.sin(i / 10000 ** (j / d))
                else:
                    result[i, j] = np.cos(i / 10000 ** (j / d))
        return result
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = x.view(-1, *self.chw)
        n, c, h, w = x.shape
        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)
        
        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) ### WRITE YOUR CODE HERE

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        out = tokens + self.positional_embeddings.unsqueeze(0).repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out


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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader,ep)


    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch: {ep} [{batch_idx * len(data)}/{len(dataloader.dataset)}'
                      f' ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
            # if batch_idx % 20 == 0:
            #     loss_dict.append(loss.item())
            # if (ep+1) % 5 == 0:
            #     torch.save(self.model.state_dict(), f'checkpoint/model_ep{ep}.pth')


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
            for data in dataloader:
                outputs = self.model(data[0])
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted)
        pred_labels = torch.cat(pred_labels)
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
    

def split_train_val(xtrain, ytrain, val_ratio=0.1):
    np.random.seed(42)
    N = xtrain.shape[0]
    Nval = int(N*val_ratio)
    indices = np.arange(xtrain.shape[0])
    np.random.shuffle(indices)

    xtrain = xtrain[indices]
    ytrain = ytrain[indices]
    xval = xtrain[:Nval]
    yval = ytrain[:Nval]
    xtrain = xtrain[Nval:]
    ytrain = ytrain[Nval:]
    
    return xtrain, ytrain, xval, yvalimport numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2
import torch.optim as optim

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
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
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = self.fc4(x)

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes,image_size=(28,28)):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN,self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after convolutional and pooling layers
        self.feature_map_size = self._get_feature_map_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * self.feature_map_size[0] * self.feature_map_size[1], 512)
        self.fc2 = nn.Linear(512, n_classes)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def _get_feature_map_size(self):
        # Create a dummy tensor with the input shape to calculate the size of the feature map
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, *self.image_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        return x.shape[2], x.shape[3]
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """      
        # Reshape the input tensor to (N, input_channels, H, W)
        x = x.view(-1, self.input_channels, *self.image_size)

        # Apply first convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply first fully connected layer with dropout and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Apply second fully connected layer
        preds = self.fc2(x)
        
        return preds

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        # assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T) / np.sqrt(self.d)                
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, n_classes=10):
        """
        Initialize the network.
        """        
        super(MyViT, self).__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, n_classes),
            nn.Softmax(dim=-1)
        )
    
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w # We assume square image.
        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches
    
    def get_positional_embeddings(self, sequence_length, d):
        """
        Get the positional embeddings for a given sequence length and hidden size.
        """
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = np.sin(i / 10000 ** (j / d))
                else:
                    result[i, j] = np.cos(i / 10000 ** (j / d))
        return result
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = x.view(-1, *self.chw)
        n, c, h, w = x.shape
        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)
        
        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) ### WRITE YOUR CODE HERE

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        out = tokens + self.positional_embeddings.unsqueeze(0).repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out


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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader,ep)


    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch: {ep} [{batch_idx * len(data)}/{len(dataloader.dataset)}'
                      f' ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
            # if batch_idx % 20 == 0:
            #     loss_dict.append(loss.item())
            # if (ep+1) % 5 == 0:
            #     torch.save(self.model.state_dict(), f'checkpoint/model_ep{ep}.pth')


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
            for data in dataloader:
                outputs = self.model(data[0])
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted)
        pred_labels = torch.cat(pred_labels)
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
    

def split_train_val(xtrain, ytrain, val_ratio=0.1):
    np.random.seed(42)
    N = xtrain.shape[0]
    Nval = int(N*val_ratio)
    indices = np.arange(xtrain.shape[0])
    np.random.shuffle(indices)

    xtrain = xtrain[indices]
    ytrain = ytrain[indices]
    xval = xtrain[:Nval]
    yval = ytrain[:Nval]
    xtrain = xtrain[Nval:]
    ytrain = ytrain[Nval:]
    
    return xtrain, ytrain, xval, yvalimport numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2
import torch.optim as optim

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
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
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = self.fc4(x)

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes,image_size=(28,28)):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN,self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after convolutional and pooling layers
        self.feature_map_size = self._get_feature_map_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * self.feature_map_size[0] * self.feature_map_size[1], 512)
        self.fc2 = nn.Linear(512, n_classes)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def _get_feature_map_size(self):
        # Create a dummy tensor with the input shape to calculate the size of the feature map
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, *self.image_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        return x.shape[2], x.shape[3]
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """      
        # Reshape the input tensor to (N, input_channels, H, W)
        x = x.view(-1, self.input_channels, *self.image_size)

        # Apply first convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply first fully connected layer with dropout and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Apply second fully connected layer
        preds = self.fc2(x)
        
        return preds

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        # assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T) / np.sqrt(self.d)                
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, n_classes=10):
        """
        Initialize the network.
        """        
        super(MyViT, self).__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, n_classes),
            nn.Softmax(dim=-1)
        )
    
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w # We assume square image.
        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches
    
    def get_positional_embeddings(self, sequence_length, d):
        """
        Get the positional embeddings for a given sequence length and hidden size.
        """
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = np.sin(i / 10000 ** (j / d))
                else:
                    result[i, j] = np.cos(i / 10000 ** (j / d))
        return result
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = x.view(-1, *self.chw)
        n, c, h, w = x.shape
        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)
        
        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) ### WRITE YOUR CODE HERE

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        out = tokens + self.positional_embeddings.unsqueeze(0).repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out


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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader,ep)


    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch: {ep} [{batch_idx * len(data)}/{len(dataloader.dataset)}'
                      f' ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
            # if batch_idx % 20 == 0:
            #     loss_dict.append(loss.item())
            # if (ep+1) % 5 == 0:
            #     torch.save(self.model.state_dict(), f'checkpoint/model_ep{ep}.pth')


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
            for data in dataloader:
                outputs = self.model(data[0])
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted)
        pred_labels = torch.cat(pred_labels)
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
    

def split_train_val(xtrain, ytrain, val_ratio=0.1):
    np.random.seed(42)
    N = xtrain.shape[0]
    Nval = int(N*val_ratio)
    indices = np.arange(xtrain.shape[0])
    np.random.shuffle(indices)

    xtrain = xtrain[indices]
    ytrain = ytrain[indices]
    xval = xtrain[:Nval]
    yval = ytrain[:Nval]
    xtrain = xtrain[Nval:]
    ytrain = ytrain[Nval:]
    
    return xtrain, ytrain, xval, yvalimport numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2
import torch.optim as optim

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
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
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = self.fc4(x)

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes,image_size=(28,28)):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN,self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after convolutional and pooling layers
        self.feature_map_size = self._get_feature_map_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * self.feature_map_size[0] * self.feature_map_size[1], 512)
        self.fc2 = nn.Linear(512, n_classes)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def _get_feature_map_size(self):
        # Create a dummy tensor with the input shape to calculate the size of the feature map
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, *self.image_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        return x.shape[2], x.shape[3]
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """      
        # Reshape the input tensor to (N, input_channels, H, W)
        x = x.view(-1, self.input_channels, *self.image_size)

        # Apply first convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply first fully connected layer with dropout and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Apply second fully connected layer
        preds = self.fc2(x)
        
        return preds

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        # assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T) / np.sqrt(self.d)                
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, n_classes=10):
        """
        Initialize the network.
        """        
        super(MyViT, self).__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, n_classes),
            nn.Softmax(dim=-1)
        )
    
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w # We assume square image.
        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches
    
    def get_positional_embeddings(self, sequence_length, d):
        """
        Get the positional embeddings for a given sequence length and hidden size.
        """
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = np.sin(i / 10000 ** (j / d))
                else:
                    result[i, j] = np.cos(i / 10000 ** (j / d))
        return result
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = x.view(-1, *self.chw)
        n, c, h, w = x.shape
        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)
        
        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) ### WRITE YOUR CODE HERE

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        out = tokens + self.positional_embeddings.unsqueeze(0).repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out


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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader,ep)


    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch: {ep} [{batch_idx * len(data)}/{len(dataloader.dataset)}'
                      f' ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
            # if batch_idx % 20 == 0:
            #     loss_dict.append(loss.item())
            # if (ep+1) % 5 == 0:
            #     torch.save(self.model.state_dict(), f'checkpoint/model_ep{ep}.pth')


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
            for data in dataloader:
                outputs = self.model(data[0])
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted)
        pred_labels = torch.cat(pred_labels)
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
    

def split_train_val(xtrain, ytrain, val_ratio=0.1):
    np.random.seed(42)
    N = xtrain.shape[0]
    Nval = int(N*val_ratio)
    indices = np.arange(xtrain.shape[0])
    np.random.shuffle(indices)

    xtrain = xtrain[indices]
    ytrain = ytrain[indices]
    xval = xtrain[:Nval]
    yval = ytrain[:Nval]
    xtrain = xtrain[Nval:]
    ytrain = ytrain[Nval:]
    
    return xtrain, ytrain, xval, yvalimport numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2
import torch.optim as optim

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
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
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = self.fc4(x)

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes,image_size=(28,28)):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN,self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after convolutional and pooling layers
        self.feature_map_size = self._get_feature_map_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * self.feature_map_size[0] * self.feature_map_size[1], 512)
        self.fc2 = nn.Linear(512, n_classes)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def _get_feature_map_size(self):
        # Create a dummy tensor with the input shape to calculate the size of the feature map
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, *self.image_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        return x.shape[2], x.shape[3]
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """      
        # Reshape the input tensor to (N, input_channels, H, W)
        x = x.view(-1, self.input_channels, *self.image_size)

        # Apply first convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply first fully connected layer with dropout and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Apply second fully connected layer
        preds = self.fc2(x)
        
        return preds

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        # assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T) / np.sqrt(self.d)                
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, n_classes=10):
        """
        Initialize the network.
        """        
        super(MyViT, self).__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, n_classes),
            nn.Softmax(dim=-1)
        )
    
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w # We assume square image.
        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches
    
    def get_positional_embeddings(self, sequence_length, d):
        """
        Get the positional embeddings for a given sequence length and hidden size.
        """
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = np.sin(i / 10000 ** (j / d))
                else:
                    result[i, j] = np.cos(i / 10000 ** (j / d))
        return result
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = x.view(-1, *self.chw)
        n, c, h, w = x.shape
        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)
        
        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) ### WRITE YOUR CODE HERE

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        out = tokens + self.positional_embeddings.unsqueeze(0).repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out


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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader,ep)


    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch: {ep} [{batch_idx * len(data)}/{len(dataloader.dataset)}'
                      f' ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
            # if batch_idx % 20 == 0:
            #     loss_dict.append(loss.item())
            # if (ep+1) % 5 == 0:
            #     torch.save(self.model.state_dict(), f'checkpoint/model_ep{ep}.pth')


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
            for data in dataloader:
                outputs = self.model(data[0])
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted)
        pred_labels = torch.cat(pred_labels)
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
    

def split_train_val(xtrain, ytrain, val_ratio=0.1):
    np.random.seed(42)
    N = xtrain.shape[0]
    Nval = int(N*val_ratio)
    indices = np.arange(xtrain.shape[0])
    np.random.shuffle(indices)

    xtrain = xtrain[indices]
    ytrain = ytrain[indices]
    xval = xtrain[:Nval]
    yval = ytrain[:Nval]
    xtrain = xtrain[Nval:]
    ytrain = ytrain[Nval:]
    
    return xtrain, ytrain, xval, yvalimport numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2
import torch.optim as optim

class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
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
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)


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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = self.fc4(x)

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes,image_size=(28,28)):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN,self).__init__()
        self.input_channels = input_channels
        self.image_size = image_size

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Calculate the size of the feature map after convolutional and pooling layers
        self.feature_map_size = self._get_feature_map_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * self.feature_map_size[0] * self.feature_map_size[1], 512)
        self.fc2 = nn.Linear(512, n_classes)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def _get_feature_map_size(self):
        # Create a dummy tensor with the input shape to calculate the size of the feature map
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, *self.image_size)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
        return x.shape[2], x.shape[3]
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """      
        # Reshape the input tensor to (N, input_channels, H, W)
        x = x.view(-1, self.input_channels, *self.image_size)

        # Apply first convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Apply second convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Apply third convolutional layer, then ReLU activation and pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Apply first fully connected layer with dropout and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Apply second fully connected layer
        preds = self.fc2(x)
        
        return preds

class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        # assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
        d_head = int(d / n_heads)
        self.d_head = d_head

        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                # Select the mapping associated to the given head.
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]

                # Map seq to q, k, v.
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T) / np.sqrt(self.d)                
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.mhsa(self.norm1(x))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, n_classes=10):
        """
        Initialize the network.
        """        
        super(MyViT, self).__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        # HINT: don't forget the classification token
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, n_classes),
            nn.Softmax(dim=-1)
        )
    
    def patchify(self, images, n_patches):
        n, c, h, w = images.shape
        assert h == w # We assume square image.
        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches
    
    def get_positional_embeddings(self, sequence_length, d):
        """
        Get the positional embeddings for a given sequence length and hidden size.
        """
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                if j % 2 == 0:
                    result[i, j] = np.sin(i / 10000 ** (j / d))
                else:
                    result[i, j] = np.cos(i / 10000 ** (j / d))
        return result
    
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = x.view(-1, *self.chw)
        n, c, h, w = x.shape
        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)
        
        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches) ### WRITE YOUR CODE HERE

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        # HINT: use torch.Tensor.repeat(...)
        out = tokens + self.positional_embeddings.unsqueeze(0).repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only.
        out = out[:, 0]

        # Map to the output distribution.
        out = self.mlp(out)

        return out


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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader,ep)


    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(dataloader):

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Epoch: {ep} [{batch_idx * len(data)}/{len(dataloader.dataset)}'
                      f' ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')
            # if batch_idx % 20 == 0:
            #     loss_dict.append(loss.item())
            # if (ep+1) % 5 == 0:
            #     torch.save(self.model.state_dict(), f'checkpoint/model_ep{ep}.pth')


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
            for data in dataloader:
                outputs = self.model(data[0])
                _, predicted = torch.max(outputs.data, 1)
                pred_labels.append(predicted)
        pred_labels = torch.cat(pred_labels)
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
    

def split_train_val(xtrain, ytrain, val_ratio=0.1):
    np.random.seed(42)
    N = xtrain.shape[0]
    Nval = int(N*val_ratio)
    indices = np.arange(xtrain.shape[0])
    np.random.shuffle(indices)

    xtrain = xtrain[indices]
    ytrain = ytrain[indices]
    xval = xtrain[:Nval]
    yval = ytrain[:Nval]
    xtrain = xtrain[Nval:]
    ytrain = ytrain[Nval:]
    
    return xtrain, ytrain, xval, yval