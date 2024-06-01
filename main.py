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

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes



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
    return xtrain, xtest, ytrain, ytest


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


def tune_cnn(xtrain, xtest, ytrain, ytest, device):
    print("[INFO] Tuning CNN hyperparameters...")
    filter_combinations = [(2, 4, 8), (4, 8, 16), (5, 10, 15), 
                           (10, 20, 30), (16, 32, 64), (32, 64, 128),
                           (64, 128, 256), (128, 256, 512)]
    accuracies = []
    f1_scores = []
    best_acc = 0
    best_f1 = 0
    best_filters = None

    t1 = time.time()

    for filters in filter_combinations:
        print(f"Training with filters: {filters}")
        model = CNN(1, get_n_classes(ytrain), conv_layers=filters).to(device)
        summary(model)

        trainer = Trainer(model, lr=1e-3, epochs=50, batch_size=64, device=device)

        s1 = time.time()
        trainer.fit(xtrain, ytrain)
        preds = trainer.predict(xtest)
        s2 = time.time()

        print(f"Time taken for prediction: {(s2 - s1) // 60} min, {(s2 - s1) % 60} sec")

        acc = accuracy_fn(preds, ytest)
        f1 = macrof1_fn(preds, ytest)
        accuracies.append(acc)
        f1_scores.append(f1)

        if acc > best_acc or (acc == best_acc and f1 > best_f1): # resolve ties by F1-score, but prioritize accuracy
            best_acc = acc
            best_f1 = f1
            best_filters = filters

    t2 = time.time()
    print(f"Total time taken for tuning: {(t2 - t1) // 60} min, {(t2 - t1) % 60} sec")
    print(f"Best filter combination: {best_filters} with accuracy: {best_acc:.3f}% and F1-score: {best_f1:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(filter_combinations)), accuracies, label='Accuracy')
    #plt.plot(range(len(filter_combinations)), f1_scores, label='F1-score')
    plt.xticks(range(len(filter_combinations)), labels=[
        '(2,4,8)', '(4,8,16)', '(5,10,15)', '(10, 20, 30)', 
        '(16,32,64)', '(32,64,128)', '(64,128,256)', '(128,256,512)'])
    plt.xlabel('Filter Combinations')
    plt.ylabel('Metrics')
    plt.title('CNN Accuracy by Filter Combinations')
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_tuning.png')
    plt.show()


def tune_transformer(xtrain, xtest, ytrain, ytest, device):
    n_blocks_options = [2, 4, 6, 8, 10]
    hidden_d_options = [128, 256, 512, 768, 1024]
    n_heads_options = [2, 4, 6, 8]
    lr_options = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    # Generate all possible combinations manually
    all_combinations = [(n_blocks, hidden_d, n_heads, lr) 
                        for n_blocks in n_blocks_options 
                        for hidden_d in hidden_d_options 
                        for n_heads in n_heads_options 
                        for lr in lr_options]
    
    # Randomly select 50 combinations
    sampled_combinations = random.sample(all_combinations, 50)

    best_acc = 0
    best_params = {}
    results = []

    t1 = time.time()

    for n_blocks, hidden_d, n_heads, lr in sampled_combinations:
        print(f"Training with n_blocks={n_blocks}, hidden_d={hidden_d}, n_heads={n_heads}, lr={lr}")
        model = MyViT(chw=(1, 28, 28), n_patches=16, n_blocks=n_blocks, 
                      hidden_d=hidden_d, n_heads=n_heads, out_d=get_n_classes(ytrain)).to(device)
        summary(model)
        trainer = Trainer(model, lr=lr, epochs=10, batch_size=64, device=device)  # Fewer epochs for faster tuning

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


def tune_mlp(xtrain, xtest, ytrain, ytest, device):
    hidden_layers_options = [1, 2, 3, 4, 5]
    hidden_units_options = [64, 128, 256, 512]
    lr_options = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    best_acc = 0
    best_params = {}
    results = []
    t1 = time.time()

    for hidden_layers in hidden_layers_options:
        for hidden_units in hidden_units_options:
            for lr in lr_options:
                print(f"Training with hidden_layers={hidden_layers}, hidden_units={hidden_units}, lr={lr}")
                model = MLP(xtrain.shape[1], get_n_classes(ytrain), hidden_units=hidden_units, hidden_layers=hidden_layers).to(device)
                summary(model)
                trainer = Trainer(model, lr=lr, epochs=10, batch_size=64, device=device)  # Fewer epochs for faster tuning

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
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    data_size = len(xtrain) + len(xtest)

    # Make a validation set
    if not args.test:
        #args.use_pca = True
        xtrain, xtest, ytrain, ytest = split_train_test(xtrain, ytrain, test_size=0.2)
        print(f"[INFO] Data loaded: xtrain.shape = {xtrain.shape} - ytrain.shape = {ytrain.shape}")
        print(f"[INFO] Data loaded: xtest.shape = {xtest.shape} - ytest.shape = {ytest.shape}")

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
                pca_obj_vis = PCA(d=3) # reduce to 3 dimensions for visualization
                xtrain_vis = pca_obj_vis.reduce_dimension(xtrain)
                xtest_vis = pca_obj_vis.reduce_dimension(xtest)
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                colors = [color for color in mcolors.CSS4_COLORS]
                rd.shuffle(colors)
                for i, point in enumerate(xtest_vis):
                    ax.scatter3D(point[0], point[1], point[2], color=colors[ytest[i]])
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
        model = MLP(xtrain.shape[1], n_classes)

    elif args.nn_type == "cnn":
        xtrain = xtrain.reshape(xtrain.shape[0], 1, 28, 28)
        xtest = xtest.reshape(xtest.shape[0], 1, 28, 28)
        if args.tune:
            tune_cnn(xtrain, xtest, ytrain, ytest, device)
            return
        best_filters = (64, 128, 256)  # using the best filter combination from tuning
        model = CNN(1, n_classes, conv_layers=best_filters)
        
    elif args.nn_type == "vit":
        if args.tune:
            tune_transformer(xtrain, xtest, ytrain, ytest, device)
            return
        model = MyViT(chw=(1, 28, 28), n_patches=16, n_blocks=6, hidden_d=512, n_heads=8, out_d=n_classes)

    summary(model)
    model = model.to(device)

    # Trainer object
    trainer = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size, device=device)

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = trainer.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = trainer.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc_train = accuracy_fn(preds_train, ytrain)
    macrof1_train = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc_train:.3f}% - F1-score = {macrof1_train:.6f}")

    if ytest is not None:
        acc_test = accuracy_fn(preds, ytest)
        macrof1_test = macrof1_fn(preds, ytest)
        print(f"Validation set: accuracy = {acc_test:.3f}% - F1-score = {macrof1_test:.6f}")

    # Save predictions
    print("\npredictions shape: \n", preds.shape)
    preds = preds.reshape(-1)  # Ensure predictions are of shape (N,)
    np.save("predictions.npy", preds)


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
    args = parser.parse_args()
    main(args)
