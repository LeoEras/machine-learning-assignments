from sklearn.ensemble import RandomForestClassifier
import sys
import pandas as pd

def train_final_model(X_train, y_train):
    model = RandomForestClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    return model

def predictMCIconverters(Xtest, data_dir):
    """
    Returns a vector of predictions with elements "0" for sMCI and "1" for pMCI,
    corresponding to each of the N_test features vectors in Xtest
    Xtest N_test x 14 matrix of test feature vectors
    data_dir full path to the folder containing the following files:
    train.fdg_pet.sMCI.csv, train.fdg_pet.pMCI.csv,
    test.fdg_pet.sMCI.csv, test.fdg_pet.pMCI.csv
    """
    # Load datasets
    train_sMCI = pd.read_csv(f"{data_dir}/train.fdg_pet.sMCI.csv", header=None)
    train_pMCI = pd.read_csv(f"{data_dir}/train.fdg_pet.pMCI.csv", header=None)
    test_sMCI = pd.read_csv(f"{data_dir}/test.fdg_pet.sMCI.csv", header=None) # unused
    test_pMCI = pd.read_csv(f"{data_dir}/test.fdg_pet.pMCI.csv", header=None) # unused

    # Prepare training data
    train_sMCI["label"] = 0
    train_pMCI["label"] = 1
    train_df = pd.concat([train_sMCI, train_pMCI], ignore_index=True)

    # Separate features and labels
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    model = train_final_model(X_train, y_train)
    ytest = model.predict(Xtest)
    return ytest

def main():
    data_directory = sys.argv[1] # Directory for the whole datasets
    Xtest_file = sys.argv[2] # For us, the second argument is the file to be used as Xtest data
    Xtest = pd.read_csv(Xtest_file)
    predictions = predictMCIconverters(Xtest, data_directory)
    print("Predictions:", predictions)

if __name__ == '__main__':
    main()