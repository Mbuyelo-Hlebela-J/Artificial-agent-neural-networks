def load_iris_dataset(filename):
    lines = None
    with open(filename,'r') as f:
        lines = f.readlines()
    inputs = []
    labels = []

    label_map ={
        "Iris-setosa":[1,0,0],
        "Iris-versicolor":[0,1,0],
        "Iris-virginica":[0,0,1]
        }

    for line in lines:
        if line.strip() == "":
            continue
        parts = line.strip().split(',')
        features =[float(x) for x in parts[:4]]
        label = label_map[parts[4]]
        inputs.append(features)
        inputs.append(label)
    return inputs, labels

def normalize_inputs(inputs):
    transposed = list(zip(*inputs))
    normalized = []

    for feature in transposed:
        min_val = min(feature)
        max_val = max(feature)
        range_val = max_val - min_val
        normalized.append([(x - min_val)/range_val for x in feature])
    return list(map(list,zip(*normalized)))
    

