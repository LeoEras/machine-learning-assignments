import pandas as pd

DATASETS = ["datasets/train.fdg_pet.sNC.csv", "datasets/train.fdg_pet.sDAT.csv", "datasets/test.fdg_pet.sNC.csv", "datasets/test.fdg_pet.sDAT.csv", "datasets/fdg_pet.feature.info.txt"]

def read_features(filename):
    features = []
    with open(filename) as features_file:
        lines = features_file.readlines()
        for line in lines:
            if "#" not in line:
                line = line.replace('\n', '')
                features.append(line)
    return features


def load_data():
    features = read_features(DATASETS[4])

    train_nc = pd.read_csv(DATASETS[0], header=None, names=features)
    train_dat = pd.read_csv(DATASETS[1], header=None, names=features)
    test_nc = pd.read_csv(DATASETS[2], header=None, names=features)
    test_dat = pd.read_csv(DATASETS[3], header=None, names=features)

    # Assign labels (0 = sNC, 1 = sDAT)
    train_nc["label"] = 0
    train_dat["label"] = 1
    test_nc["label"] = 0
    test_dat["label"] = 1

    # Merge train and test data
    train_data = pd.concat([train_nc, train_dat], ignore_index=True)
    test_data = pd.concat([test_nc, test_dat], ignore_index=True)

    # Separate features and labels
    X_train = train_data.drop(columns=["label"])
    y_train = train_data["label"]
    X_test = test_data.drop(columns=["label"])
    y_test = test_data["label"]

    return X_train, y_train, X_test, y_test