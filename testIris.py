from genetic_algorithm import *
import numpy as np

def readData(filename):
    f = open(filename, 'r')
    data = f.readlines()

    input_data = []
    for i in range(len(data)):
        s = data[i].split()
        input_data.append(s)

    return np.array(input_data)


# ------------------- Iris Dataset ------------------- #

iris_train = readData("./iris_data/iris-train.txt")
iris_test = readData("./iris_data/iris-test.txt")

# Data prep
iris_train_y = np.array(iris_train)[:, -1]
iris_train_X = np.array(iris_train)[:, :-1]

iris_test_y = np.array(iris_test)[:, -1]
iris_test_X = np.array(iris_test)[:, :-1]

iris_train_X = np.int32(iris_train_X.astype(np.float) * 10)
iris_test_X = np.int32(iris_test_X.astype(np.float) * 10)

iris_head = "sepal_length", "sepal_width", "petal_length", "petal_width", "Iris"

target_bits = 2
model = Genetic(iris_train_X.copy(), iris_train_y.copy(), 
                    iris_test_X.copy(), iris_test_y.copy(), 
                    iris_head, target_bits, population_len=500,
                    fitness_threshold=1, minmax=True, 
                    max_allowed_rules=5, generation_stop=150, plotOn=False)

model.print_data()
model.train()

train_acc = model.accuracy(iris_train_X.copy(), iris_train_y.copy())
test_acc = model.accuracy(iris_test_X.copy(), iris_test_y.copy())

print(f"Train Accuracy: {train_acc} | Test Accuracy: {test_acc}")

