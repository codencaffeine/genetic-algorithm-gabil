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

selections = ["Rank", "Roulette", "Tournament"]
train_acc_logs = []
test_acc_logs = []
accuracy_list = []

r_list = np.arange(0.1, 1.0, 0.1)
for r in r_list:
    print(f"\n\nReplacement Rate:{r}")
    model = Genetic(iris_train_X.copy(), iris_train_y.copy(), 
                    iris_test_X.copy(), iris_test_y.copy(), 
                    iris_head, target_bits, population_len=500,
                    minmax=True, generation_stop=150,
                    replacement_rate=r, max_allowed_rules=5, plotOn=False)

    train_log, test_log = model.train()
    train_acc = model.accuracy(iris_train_X.copy(), iris_train_y.copy())
    test_acc = model.accuracy(iris_test_X.copy(), iris_test_y.copy())
    accuracy_list.append("Train Accuracy: " + str(train_acc) + " | Test Accuracy: " + str(test_acc))
    
    train_acc_logs.append(train_log[-1])
    test_acc_logs.append(test_log[-1])


for r in range(len(train_acc_logs)):
    print("r:", r_list[r], accuracy_list[r])
    
if model.plotOn:
    import matplotlib.pyplot as plt
    plt.plot(r_list, test_acc_logs)
        
    plt.xlabel("Replacement Rate")
    plt.ylabel("Test Accuracy")
    plt.title("Replacement Rate vs Test Accuracy")
    plt.grid()
    plt.show()