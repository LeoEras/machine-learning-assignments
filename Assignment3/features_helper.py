
def read_features(filename):
    features = []
    with open(filename) as features_file:
        lines = features_file.readlines()
        for line in lines:
            if "#" not in line:
                line = line.replace('\n', '')
                features.append(line)
    return features