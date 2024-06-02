import argparse
import numpy as np
import torch
from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits import mplot3d
import random
import random as rd
import time
import os

from torchvision import transforms  # for data augmentation

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

# =============================================================================
# ====================== Data augmentation for CNN ============================
# =============================================================================

# Data augmentation for CNN
transform_train = transforms.Compose([
     transforms.ToPILImage(),                # converts tensor or ndarray to PIL Image.
    transforms.RandomHorizontalFlip(),       # randomly flip the image horizontally with probability 1/2
    transforms.RandomRotation(10),           # randomly rotate image by a value between -10 and 10 degrees.
    transforms.RandomCrop(28, padding=4),    
    transforms.ToTensor(),                   # converts PIL Image or numpy array to tensor.
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

def load_data_with_augmentation(data_dir, transform=None):
    x = np.load(os.path.join(data_dir, 'train_data.npy'))
    y = np.load(os.path.join(data_dir, 'train_label.npy'))
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    if transform:
        x_augmented = torch.stack([transform(img.numpy().reshape(28, 28, 1)) for img in x_tensor])
        return x_augmented, y_tensor.numpy()  
    else:
        return x_tensor, y_tensor.numpy()  

# ===================================================================================================================================
# ===================================================================================================================================
# ===================================================================================================================================
    
def split_train_test(x, y, test_size=0.2, random_seed=None):
    """
    Split the data into training and testing sets.
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    indices = np.random.permutation(len(x))
    test_size = int(len(x) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    xtrain, xtest = x[train_indices], x[test_indices]
    ytrain, ytest = y[train_indices], y[test_indices]
    print(f"[INFO] Data loaded: xtest.shape = {xtest.shape} - ytest.shape = {ytest.shape}")
    print(f"[INFO] Data split: xtrain.shape = {xtrain.shape} - ytrain.shape = {ytrain.shape}")
    print(f"[INFO] Data split percentages: {len(xtrain) / (len(xtrain) + len(xtest)):.2f} [train] - {len(xtest) / (len(xtrain) + len(xtest)):.2f} [test]")
    return xtrain, xtest, ytrain, ytest

# ===================================================================================================================================
# ===================================================================================================================================
# ===================================================================================================================================

def tune_pca(xtrain, xtest, ytrain, ytest, device):
    """
    Tune the number of PCA components to use for dimensionality reduction.
    """

    print("[INFO] Tuning PCA components...")
    pca_components = list(range(1, xtrain.shape[1], 35)) # test every 35th component
    variances = []
    accuracies = []
    best_acc = 0
    best_d = None

    t1 = time.time()

    for d in pca_components:
        pca_obj = PCA(d=d)
        pca_obj.find_principal_components(xtrain)
        xtrain_reduced = pca_obj.reduce_dimension(xtrain)
        xtest_reduced = pca_obj.reduce_dimension(xtest)

        model = MLP(xtrain_reduced.shape[1], get_n_classes(ytrain)).to(device)
        summary(model)

        trainer = Trainer(model, lr=1e-3, epochs=50, batch_size=64, device=device)
        
        s1 = time.time()
        trainer.fit(xtrain_reduced, ytrain)
        preds = trainer.predict(xtest_reduced)
        s2 = time.time()

        print(f"Time taken for training: {(s2 - s1) // 60} min, {(s2 - s1) % 60} sec")

        acc = accuracy_fn(preds, ytest)
        
        variances.append(np.sum(pca_obj.explained_variance_ratio_))
        accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_d = d

    t2 = time.time()
    print(f"Total time taken for tuning: {(t2 - t1) // 60} min, {(t2 - t1) % 60} sec")
    print(f"Best number of PCA components: {best_d} with accuracy: {best_acc:.3f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(pca_components, variances, label='Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance')
    plt.title('PCA Variance Explained by Number of Components')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_variance.png')
    plt.show()

# ===================================================================================================================================
# ===================================================================================================================================
# ===================================================================================================================================
        
def tune_cnn(xtrain, xtest, ytrain, ytest, device):
    print("[INFO] Tuning CNN hyperparameters...")
    filter_combinations = [(2, 4, 8), (4, 8, 16), (5, 10, 15), 
                           (10, 20, 30), (16, 32, 64), (32, 64, 128),
                           (64, 128, 256), (128, 256, 512)]
    lr_options = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    iterations_options = [10, 20, 50]
    
    accuracies = []
    f1_scores = []
    best_acc = 0
    best_f1 = 0
    best_filters = None
    best_lr = None
    best_iters = None
    iteration_nb = 0

    t1 = time.time()

    all_combinations = [(filters, lr, iters) for filters in filter_combinations for lr in lr_options for iters in iterations_options]
    sampled_combinations = random.sample(all_combinations, 20)  # sample 20 random combinations

    for filters, lr, iters in sampled_combinations:
        iteration_nb += 1
        print(f"Iteration #", iteration_nb)
        print(f"Training with filters={filters}, lr={lr}, iters={iters}")
        model = CNN(1, get_n_classes(ytrain), conv_layers=filters).to(device)
        summary(model)

        trainer = Trainer(model, lr=lr, epochs=iters, batch_size=64, device=device)

        s1 = time.time()
        trainer.fit(xtrain, ytrain)
        preds = trainer.predict(xtest)
        s2 = time.time()

        print(f"Time taken for prediction: {(s2 - s1) // 60} min, {(s2 - s1) % 60} sec")

        acc = accuracy_fn(preds, ytest)
        f1 = macrof1_fn(preds, ytest)
        accuracies.append(acc)
        f1_scores.append(f1)

        print(f"Accuracy: {acc:.3f}% - F1-score: {f1:.6f}")

        if acc > best_acc or (acc == best_acc and f1 > best_f1):  # Resolve ties by F1-score, but prioritize accuracy
            best_acc = acc
            best_f1 = f1
            best_filters = filters
            best_lr = lr
            best_iters = iters

    t2 = time.time()
    print(f"Total time taken for tuning: {(t2 - t1) // 60} min, {(t2 - t1) % 60} sec")
    print(f"Best filter combination: {best_filters} with lr: {best_lr} and max_iters: {best_iters} and accuracy: {best_acc:.3f}% and F1-score: {best_f1:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sampled_combinations)), accuracies, label='Accuracy')
    #plt.plot(range(len(filter_combinations)), f1_scores, label='F1-score')
    plt.xticks(range(len(sampled_combinations)), labels=[
        f'{filters}-{lr}-{iters}' for filters, lr, iters in sampled_combinations], rotation=90)
    plt.xlabel('Filter-LR-Iters Combinations')
    plt.ylabel('Metrics')
    plt.title('CNN Accuracy by Filter-LR-Iters Combinations')
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_tuning.png')
    plt.show()

# ===================================================================================================================================
# ===================================================================================================================================
# ===================================================================================================================================
        
def tune_transformer(xtrain, xtest, ytrain, ytest, device):
    print("[INFO] Tuning Transformer hyperparameters...")
    n_blocks_options = [2, 4, 6, 8, 10]
    hidden_d_options = [128, 256, 512, 768, 1024]
    n_heads_options = [2, 4, 6, 8]
    lr_options = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    iteration = 0

    all_combinations = [(n_blocks, hidden_d, n_heads, lr) 
                        for n_blocks in n_blocks_options 
                        for hidden_d in hidden_d_options 
                        for n_heads in n_heads_options 
                        for lr in lr_options]
    
    sampled_combinations = random.sample(all_combinations, 25) # try 25 random sample combinations for tuning 
    best_acc = 0
    best_params = {}
    results = []

    t1 = time.time()

    for n_blocks, hidden_d, n_heads, lr in sampled_combinations:
        iteration += 1
        print(f"Iteration #", iteration)
        print(f"Training with n_blocks={n_blocks}, hidden_d={hidden_d}, n_heads={n_heads}, lr={lr}")
        model = MyViT(chw=(1, 28, 28), n_patches=16, n_blocks=n_blocks, 
                      hidden_d=hidden_d, n_heads=n_heads, out_d=get_n_classes(ytrain)).to(device)
        summary(model)
        trainer = Trainer(model, lr=lr, epochs=10, batch_size=64, device=device)  

        s1 = time.time()
        trainer.fit(xtrain, ytrain)
        preds = trainer.predict(xtest)
        s2 = time.time()

        print(f"Time taken for training: {(s2 - s1) // 60} min, {(s2 - s1) % 60} sec")

        acc = accuracy_fn(preds, ytest)
        results.append((n_blocks, hidden_d, n_heads, lr, acc))

        if acc > best_acc:
            best_acc = acc
            best_params = {'n_blocks': n_blocks, 'hidden_d': hidden_d, 'n_heads': n_heads, 'lr': lr}

    t2 = time.time()
    print(f"Total time taken for tuning: {(t2 - t1) // 60} min, {(t2 - t1) % 60} sec")
    print(f"Best hyperparameters: {best_params} with accuracy: {best_acc:.3f}%")

    plt.figure(figsize=(10, 5))
    for result in results:
        plt.scatter(result[1], result[4], label=f"n_blocks={result[0]}, n_heads={result[2]}, lr={result[3]}")
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Accuracy')
    plt.title('Transformer Tuning Results')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig('transformer_tuning.png')
    plt.show()

# ===================================================================================================================================
# ===================================================================================================================================
# ===================================================================================================================================
        
def tune_mlp(xtrain, xtest, ytrain, ytest, device):
    print("[INFO] Tuning MLP hyperparameters...")
    hidden_layers_options = [1, 2, 3, 4, 5]
    hidden_units_options = [64, 128, 256, 512]
    lr_options = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    all_combinations = [(hidden_layers, hidden_units, lr) 
                        for hidden_layers in hidden_layers_options 
                        for hidden_units in hidden_units_options 
                        for lr in lr_options]
    

    sampled_combinations = random.sample(all_combinations, 20) # sample 20 random combinations 
    
    best_acc = 0
    best_params = {}
    results = []
    iteration = 0
    t1 = time.time()

    for hidden_layers, hidden_units, lr in sampled_combinations:
        iteration += 1
        print(f"Iteration #", iteration)
        print(f"Training with hidden_layers={hidden_layers}, hidden_units={hidden_units}, lr={lr}")
        model = MLP(xtrain.shape[1], get_n_classes(ytrain), hidden_units=hidden_units, hidden_layers=hidden_layers).to(device)
        summary(model)
        trainer = Trainer(model, lr=lr, epochs=10, batch_size=64, device=device) # using 10 epochs for faster training 

        s1 = time.time()
        trainer.fit(xtrain, ytrain)
        preds = trainer.predict(xtest)
        s2 = time.time()

        print(f"Time taken for training: {(s2 - s1) // 60} min, {(s2 - s1) % 60} sec")

        acc = accuracy_fn(preds, ytest)
        results.append((hidden_layers, hidden_units, lr, acc))

        if acc > best_acc:
            best_acc = acc
            best_params = {'hidden_layers': hidden_layers, 'hidden_units': hidden_units, 'lr': lr}

    t2 = time.time()
    print(f"Total time taken for tuning: {(t2 - t1) // 60} min, {(t2 - t1) % 60} sec")
    print(f"Best hyperparameters: {best_params} with accuracy: {best_acc:.3f}%")

    plt.figure(figsize=(10, 5))
    for result in results:
        plt.scatter(result[1], result[3], label=f"hidden_layers={result[0]}, lr={result[2]}")
    plt.xlabel('Hidden Units')
    plt.ylabel('Accuracy')
    plt.title('MLP Tuning Results')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig('mlp_tuning.png')
    plt.show()

# ===================================================================================================================================
# ===================================================================================================================================
# ===================================================================================================================================      


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """

    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"[INFO] Using device: {device}")

    ## 1. First, we load our data and flatten the images into vectors
    if args.nn_type == "cnn" and args.augment_data:
        print("[INFO] Using data augmentation for CNN")
        x, y = load_data_with_augmentation(args.data, transform=transform_train)
        xtrain, xtest, ytrain, ytest = split_train_test(x, y, test_size=0.2)  # Split data after augmentation
        xtrain = xtrain.reshape(xtrain.shape[0], 1, 28, 28)
        xtest = xtest.reshape(xtest.shape[0], 1, 28, 28)
        print(f"[INFO] Augmented training data shape: xtrain = {xtrain.shape} - ytrain = {ytrain.shape}")
    else:
        xtrain, xtest, ytrain = load_data(args.data)
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
        data_size = len(xtrain) + len(xtest)

    # Make a validation set
    if not args.test:
        #args.use_pca = True
        xtrain, xtest, ytrain, ytest = split_train_test(xtrain, ytrain, test_size=0.2)

    else:
        ytest = None

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
        
    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("[INFO] Using PCA for feature reduction")
        pca_d = 106 if not args.tune else args.pca_d # using 106 as the default value for PCA components (see PCA tuning results)
        if args.tune:
            tune_pca(xtrain, xtest, ytrain, ytest, device)
        else:
            pca_obj = PCA(d=pca_d)
            pca_obj.find_principal_components(xtrain)
            xtrain = pca_obj.reduce_dimension(xtrain)
            xtest = pca_obj.reduce_dimension(xtest)

            if args.visualize:
                pca_obj_vis = PCA(d=3)
                pca_obj_vis.find_principal_components(xtrain)  # Compute principal components for visualization
                xtrain_vis = pca_obj_vis.reduce_dimension(xtrain)
                xtest_vis = pca_obj_vis.reduce_dimension(xtest)
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                colors = [color for color in mcolors.CSS4_COLORS]
                rd.shuffle(colors)
                for i, point in enumerate(xtest_vis):
                    ax.scatter3D(point[0], point[1], point[2], color=colors[ytest[i]])
                plt.title('PCA Visualization of Test Data')
                plt.savefig('pca_visualization.png')
                plt.show()


    ## 3. Initialize the method you want to use.
    
    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)

    if args.nn_type == "mlp":
        if args.tune:
            tune_mlp(xtrain, xtest, ytrain, ytest, device)
            return  
        best_params = {'hidden_units': 512, 'hidden_layers': 4, 'lr': 0.0001} # using the best hyperparameters from tuning (85.442% accuracy)
        model = MLP(xtrain.shape[1], n_classes, hidden_units=best_params['hidden_units'], hidden_layers=best_params['hidden_layers'])

    elif args.nn_type == "cnn":
        xtrain = xtrain.reshape(xtrain.shape[0], 1, 28, 28)
        xtest = xtest.reshape(xtest.shape[0], 1, 28, 28)
        if args.tune:
            tune_cnn(xtrain, xtest, ytrain, ytest, device)
            return
        best_filters = (64, 128, 256)  # using the best filter combination from tuning
        best_params = {'lr': 0.0005, 'max_iters': 20}  # using the best learning rate and max_iters from tuning (89.967% accuracy)
        model = CNN(1, n_classes, conv_layers=best_filters, dropout_prob=args.dropout_prob)
        print(f"[INFO] CNN model initialized with dropout probability : {args.dropout_prob}")
        
    elif args.nn_type == "vit":
        if args.tune:
            tune_transformer(xtrain, xtest, ytrain, ytest, device)
            return
        best_params = {'n_blocks': 2, 'hidden_d': 1024, 'n_heads': 4, 'lr': 0.0001} # using the best hyperparameters from tuning (83.1% accuracy)
        model = MyViT(chw=(1, 28, 28), n_patches=16, n_blocks=best_params['n_blocks'], 
                  hidden_d=best_params['hidden_d'], n_heads=best_params['n_heads'], out_d=n_classes)

    summary(model)
    model = model.to(device)

    t1 = time.time()

    # Trainer object
    trainer = Trainer(model, lr=best_params['lr'] if args.nn_type in ['mlp', 'cnn', 'vit'] else args.lr, 
                  epochs=best_params['max_iters'] if args.nn_type == 'cnn' else args.max_iters, 
                  batch_size=args.nn_batch_size, device=device)

    ## 4. Train and evaluate the method

    # ensure xtrain is a NumPy array
    if torch.is_tensor(xtrain):
        xtrain = xtrain.numpy()

    # Fit (:=train) the method on the training data
    preds_train = trainer.fit(xtrain, ytrain)

    # ensure xtest is a NumPy array
    if torch.is_tensor(xtest):
        xtest = xtest.numpy()


    # Predict on unseen data
    preds = trainer.predict(xtest)

    t2 = time.time()
    print(f"Time taken for training: {(t2 - t1) // 60} min, {(t2 - t1) % 60} sec")

    ## Report results: performance on train and valid/test sets
    acc_train = accuracy_fn(preds_train, ytrain)
    macrof1_train = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc_train:.3f}% - F1-score = {macrof1_train:.6f}")

    if ytest is not None:
        acc_test = accuracy_fn(preds, ytest)
        macrof1_test = macrof1_fn(preds, ytest)
        print(f"Validation set: accuracy = {acc_test:.3f}% - F1-score = {macrof1_test:.6f}")

    # Save predictions
    print(f"[INFO] predictions shape: {preds.shape}")
    if preds.shape[0] > 10000:
        print("[INFO] prediction size [0] exceeds 10'000 : cropping predictions...")
        preds = preds[:10000]
        print(f"[INFO] New predictions shape: {preds.shape}")

    preds = preds.reshape(-1)  # Ensure predictions are of shape (N,)
    np.save(args.preds_file, preds)
    print(f"[INFO] Predictions saved to '{args.preds_file}'")


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="Path to your dataset")
    parser.add_argument('--nn_type', default="mlp", help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu", help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    
    # Added arguments 
    parser.add_argument('--tune', action="store_true", help="Tune hyperparameters for PCA or CNN")
    parser.add_argument('--visualize', action="store_true", help="Visualize PCA results in 3D") 
    parser.add_argument('--dropout_prob', type=float, default=0.25, help="dropout probability for CNN")
    parser.add_argument('--augment_data', action="store_true", help="use data augmentation for CNN")
    parser.add_argument('--preds_file', type=str, default="predictions.npy", help="File in which to save predictions")


    args = parser.parse_args()
    main(args)
