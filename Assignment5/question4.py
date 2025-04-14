from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from utils import data_split, img_to_arr, evaluate_model
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
import torch

# This generates the layers
def generate_random_layer_configs(n_configs=5, max_layers=10, min_neurons=10, max_neurons=100):
    configs = []
    for _ in range(n_configs):
        n_layers = np.random.randint(1, max_layers + 1)
        config = tuple(np.random.randint(min_neurons, max_neurons + 1) for _ in range(n_layers))
        configs.append(config)
    return configs

def MLPSearch(X_train, y_train, halve=False):
    print("\nTraining MLP model using Grid Search...")

    h_l_s = generate_random_layer_configs(50, 8, 10, 100)

    param_grid = {
        'hidden_layer_sizes': h_l_s
    }

    if halve: # To half size, otherwise it takes 4 hours to finish (13 min per pass, 18 passes)
        X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=.5, random_state=0, stratify=y_train)
        # Takes way less time, about an hour!
        # Pretty cool, huh?

    svm = MLPClassifier(verbose=True)

    # Cross-validation grid search (Reduced cv=3 for speed)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring="accuracy", verbose=3)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_hls = best_params["hidden_layer_sizes"]
    print(f"Best parameters found: C = {best_hls}")

    return best_hls

def train_final_model(X_train, y_train, hidden_layer_sizes):
    print("\nTraining final MLP model with best parameters...")
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    model.fit(X_train, y_train)
    print("Final model training complete!")
    return model

class MLPClassifierTorch(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes):
        super(MLPClassifierTorch, self).__init__()

        layers = []

        layer_sizes = [input_size] + list(hidden_layer_sizes) + [output_size]

        for in_size, out_size in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())

        # Final layer (no activation — assume you're using CrossEntropyLoss which includes softmax)
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = data_split()
    X_train_arr = img_to_arr(X_train)
    X_test_arr = img_to_arr(X_test)
    # hidden_layer_sizes = MLPSearch(X_train_arr, y_train, False)
    hidden_layer_sizes = (72, 62, 62, 72, 23, 60, 36, 61, 54, 98, 41, 33, 62, 71, 53, 11) # After 45 mins training
    model = train_final_model(X_train_arr, y_train, hidden_layer_sizes=hidden_layer_sizes)
    evaluate_model(model, X_test_arr, y_test, "MLP", "micro")

    # The model but as a Torch one to be saved
    model = MLPClassifierTorch(784, 10, hidden_layer_sizes) # 784=28x28, 10 (0-9)
    torch.save(model.state_dict(), "model.pth")