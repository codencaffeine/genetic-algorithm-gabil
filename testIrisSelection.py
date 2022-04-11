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

for s in selections:
    print(f"\n\nRunning {s} selection strategy...")
    model = Genetic(iris_train_X.copy(), iris_train_y.copy(), 
                    iris_test_X.copy(), iris_test_y.copy(), 
                    iris_head, target_bits, population_len=500,
                    generation_stop=5, minmax=True, 
                    selection=s, max_allowed_rules=5, plotOn=False)

    train_log, test_log = model.train()
    train_acc = model.accuracy(iris_train_X.copy(), iris_train_y.copy())
    test_acc = model.accuracy(iris_test_X.copy(), iris_test_y.copy())
    accuracy_list.append("Train Accuracy: " + str(train_acc) + " | Test Accuracy: " + str(test_acc))
    
    train_acc_logs.append(train_log)
    test_acc_logs.append(test_log)

    
for s in range(len(selections)):
    print(selections[s], "Selection: ", accuracy_list[s])
    
    if model.plotOn:
        import matplotlib.pyplot as plt
        plt.plot(test_acc_logs[s], label=selections[s])
        
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.title("Generation vs Accuracy for different selection strategies")
        plt.legend()
        plt.grid()
        plt.show()
                
    # np.savetxt(selections[s] + "_run.csv", np.array(all_run_logs[s]), delimiter=",")
    # np.savetxt(selections[s] + "_best.csv", np.array(all_best_logs[s]), delimiter=",")
    


