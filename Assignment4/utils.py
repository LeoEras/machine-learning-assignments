import pandas as pd

DATASETS = ["datasets/train.fdg_pet.sMCI.csv", "datasets/train.fdg_pet.pMCI.csv", "datasets/test.fdg_pet.sMCI.csv", "datasets/test.fdg_pet.pMCI.csv", "datasets/fdg_pet.feature.info1.txt"]

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
    train_sMCI = pd.read_csv(DATASETS[0], header=None)
    train_pMCI = pd.read_csv(DATASETS[1], header=None)
    test_sMCI = pd.read_csv(DATASETS[2], header=None)
    test_pMCI = pd.read_csv(DATASETS[3], header=None)
    features = read_features(DATASETS[4])
    
    # Setting the column name
    train_sMCI.columns = features
    train_pMCI.columns = features
    test_sMCI.columns = features
    test_pMCI.columns = features

    # Setting some outcome labels
    test_sMCI["label"] = 0
    test_pMCI["label"] = 1
    train_sMCI["label"] = 0
    train_pMCI["label"] = 1

    # Joining the blobs
    train_df = pd.concat([train_sMCI, train_pMCI], ignore_index=True)
    test_df = pd.concat([test_sMCI, test_pMCI], ignore_index=True)

    # Separate features and labels
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    return X_train, y_train, X_test, y_test