import argparse
import numpy as np
import torch
from torchinfo import summary
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits import mplot3d
import random as rd

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes


def split_train_test(x, y, test_size=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    indices = np.random.permutation(len(x))
    test_size = int(len(x) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    xtrain, xtest = x[train_indices], x[test_indices]
    ytrain, ytest = y[train_indices], y[test_indices]
    return xtrain, xtest, ytrain, ytest

def main(args):
    """
    Main function to load data, preprocess it, and train a model based on the provided arguments.

    Arguments:
        args (Namespace): Command line arguments.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"\N[INFO] Using device: {device}\n")
    # 1. Load and preprocess data
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    data_size = len(xtrain) + len(xtest)

    # Create a validation set if not testing
    if not args.test:
        xtrain, xtest, ytrain, ytest = split_train_test(xtrain, ytrain, test_size=0.2)
        print(f"[INFO] Data loaded: xtrain.shape = {xtrain.shape} - ytrain.shape = {ytrain.shape}")
        print(f"[INFO] Data loaded: xtest.shape = {xtest.shape} - ytest.shape = {ytest.shape}")
    else:
        ytest = None

    # 2. Dimensionality reduction with PCA if specified
    if args.use_pca:
        pca_obj = PCA(d=args.pca_d)
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)

        if args.visualize_pca:
            pca_obj_vis = PCA(d=3)
            xtrain_vis = pca_obj_vis.reduce_dimension(xtrain)
            xtest_vis = pca_obj_vis.reduce_dimension(xtest)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            colors = [color for color in mcolors.CSS4_COLORS]
            rd.shuffle(colors)
            for i, point in enumerate(xtest_vis):
                ax.scatter3D(point[0], point[1], point[2], color=colors[ytest[i]])
            plt.show()

    # 3. Initialize the model
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(xtrain.shape[1], n_classes)
    elif args.nn_type == "cnn":
        xtrain = xtrain.reshape(xtrain.shape[0], 1, 28, 28)
        xtest = xtest.reshape(xtest.shape[0], 1, 28, 28)
        model = CNN(1, n_classes)
    elif args.nn_type == "vit":
        model = MyViT(chw=(1, 28, 28), n_patches=16, n_blocks=6, hidden_d=512, n_heads=8, out_d=n_classes)

    summary(model)

    # 4. Train and evaluate the model
    trainer = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
    preds_train = trainer.fit(xtrain, ytrain)
    preds = trainer.predict(xtest)

    # Report results
    acc_train = accuracy_fn(preds_train, ytrain)
    macrof1_train = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc_train:.3f}% - F1-score = {macrof1_train:.6f}")

    if ytest is not None:
        acc_test = accuracy_fn(preds, ytest)
        macrof1_test = macrof1_fn(preds, ytest)
        print(f"Validation set: accuracy = {acc_test:.3f}% - F1-score = {macrof1_test:.6f}")
    
    # Save predictions
    np.save("predictions", preds.numpy())


    # Optional additional outputs or visualizations can be added here

if __name__ == '__main__':
    # Argument parsing for command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset", type=str, help="Path to your dataset")
    parser.add_argument('--nn_type', default="mlp", help="Network type: 'mlp', 'cnn', or 'vit'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="Maximum iterations for training")
    parser.add_argument('--test', action="store_true", help="Evaluate on test data")
    parser.add_argument('--use_pca', action="store_true", help="Use PCA for dimensionality reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="Number of PCA components")
    parser.add_argument('--visualize_pca', action="store_true", help="Visualize PCA results in 3D")

    args = parser.parse_args()
    main(args)
