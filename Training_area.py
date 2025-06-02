import neuron_network as net
import iris_loader as iris
import random


data = []
with open("iris.data.txt") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        features = list(map(float,parts[:4]))
        label = parts[4]

        if label == "Iris-setosa":
            target= [1,0,0]
        elif label == "Iris-versicolor":
            target = [0,1,0]
        else:
            target = [0,0,1]
        data.append((features,target))
    inputs, targets = zip(*data)


inputs, targets = zip(*data)

split = int(0.8 * len(inputs))
train_inputs, test_inputs = inputs[:split],inputs[split:]
train_targets,test_targets = targets[:split],targets[split:]
iris_classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']
nn = net.NeuralNetwork(input_size=4,hidden_size=5,output_size=3,hidden_activation="relu",output_activation="softmax")

for epoch in range(3000):
    nn.train_batch(test_inputs,test_targets)

correct = 0
for x,y in zip(test_inputs,test_targets):
    output = nn.forward(x)
    predicted= output.index(max(output))
    actual = y.index(max(y))
    pred_class = iris_classes[predicted]
    if predicted == actual:
        correct += 1

    true_index = y.index(1)
    true_class = iris_classes[true_index]
    print(f"Predicted:{pred_class},Actual:{true_class}")
    
        

