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


# ------------------- Tennis Dataset ------------------- #
print("\n\n********* Tennis Dataset ********")
tennis_train = readData("./tennis_data/tennis-train.txt")
tennis_test = readData("./tennis_data/tennis-test.txt")

tennis_train_y = np.array(tennis_train)[:, -1]
tennis_train_X = np.array(tennis_train)[:, :-1]

tennis_test_y = np.array(tennis_test)[:, -1]
tennis_test_X = np.array(tennis_test)[:, :-1]


tennis_head = ["Outlook","Temperature","Humidity","Wind","PlayTennis"]

target_bits = 1

model = Genetic(tennis_train_X.copy(), tennis_train_y.copy(), 
                tennis_test_X.copy(), tennis_test_y.copy(), 
                tennis_head, target_bits, population_len=10,
                fitness_threshold=1, max_allowed_rules=3, plotOn=False)

# model = Genetic(tennis_train_X.copy(), tennis_train_y.copy(), 
#                 tennis_test_X.copy(), tennis_test_y.copy(), 
#                 tennis_head, target_bits, population_len=10,
#                 fitness_threshold=1, max_allowed_rules=3, selection="Tournament")

# model = Genetic(tennis_train_X.copy(), tennis_train_y.copy(), 
#                 tennis_test_X.copy(), tennis_test_y.copy(), 
#                 tennis_head, target_bits, population_len=10,
#                 fitness_threshold=1, max_allowed_rules=3, selection="Rank")

model.print_data()
model.train()

accuracy = model.accuracy(tennis_train_X.copy(), tennis_train_y.copy())
print("Train accuracy:", accuracy)

accuracy = model.accuracy(tennis_test_X.copy(), tennis_test_y.copy())
print("Test accuracy:", accuracy)