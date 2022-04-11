import math
import numpy as np
import random
import itertools
import time

# random.seed(415)

class Genetic:
    def __init__(self, trainX, trainy, testX, testy, head, target_bits, 
                 population_len=100, replacement_rate=0.6, 
                 mutation_rate=0.3, fitness_threshold=1, 
                 generation_stop=None, selection="Roulette", 
                 minmax=False, variable_length=True, max_allowed_rules=4, plotOn=False):
        """initializes the algorithm parameters

        Args:
            trainX (array): Train data (attributes) as np array
            trainy (array): Train targets
            testX (array): Test data (attributes) as np array
            testy (array): Train
            head (list): Attribute name list
            target_bits (int): No of bits needed to represent target value
            population_len (int, optional): Population size. Defaults to 100.
            replacement_rate (float, optional): Replacement rate. Defaults to 0.6.
            mutation_rate (float, optional): Mutation rate. Defaults to 0.3.
            fitness_threshold (int, optional): Target Fitness threshold (stopping criteria). Defaults to 1.
            generation_stop (_type_, optional): Run GA for n generations (stopping criteria). Defaults to None.
            selection (str, optional): Selection Strategy. Defaults to "Roulette".
            minmax (bool, optional): Flag to represent continuous (interval) data. Set True for Iris dataset. Defaults to False.
            variable_length (bool, optional): Flag to enable variable length hypothesis. Defaults to "True".
            max_allowed_rules (int, optional): Max. allowed rules per hypothesis. Defaults to 4.
            plotOn (bool, optional): Enable plotting (requires Matplotlib). Defaults to "False".
            
        """
        self.data = trainX
        self.target = trainy
        self.testX = testX
        self.testy = testy
        self.p = population_len
        self.m = mutation_rate
        self.r = replacement_rate
        self.fitness_threshold = fitness_threshold
        self.selection = selection
        self.best_hypothesis = None
        self.minmax = minmax
        self.generation_stop = generation_stop
        self.plotOn = plotOn
        self.max_allowed_rules = max_allowed_rules
        
        if variable_length:
            self.variable_len_h = 2
        else:
            self.variable_len_h = 1
        
        replacement_size = math.ceil(self.r * self.p)
        # Make sure replacement_size is even number
        self.replacement_size = replacement_size - 1 if replacement_size % 2 else replacement_size
        self.keep_size = self.p - self.replacement_size
        
        self.attributes = head[:-1]
        self.target_name = head[-1]
        self.classes = np.unique(self.target)
        
        # target encoding
        self.target_bits = target_bits
        no_classes = len(self.classes)
        bin_encode = np.unpackbits(
            np.arange(0, no_classes, dtype=np.uint8)).reshape(-1,8)[:, -target_bits:]

        self.target_encode = {}
        for i in range(no_classes):
            self.target_encode[self.classes[i]] = "".join(map(str,bin_encode[i]))
        
        self.target_encode_inv = dict(zip(self.target_encode.values(), self.target_encode.keys()))

        self.sample_size = self.data.shape[0]
        self.attribute_vals = {}
        self.individual_size = 0
        
        for a in range(len(self.attributes)):
            self.attribute_vals[self.attributes[a]] = np.unique(self.data[:, a], axis=0)
        
        if self.minmax:
            self.encodings, self.bits_needed = self.generateEncodings(self.data)
            # self.encodings_inv = dict(zip(self.encodings.values(), self.encodings.keys()))
            self.individual_size = np.sum(self.bits_needed)*2 + self.target_bits
            # print(self.individual_size, self.bits_needed)
            self.member_indices = np.vstack((self.bits_needed, self.bits_needed)).flatten('F')
        
        else:
            for a in range(len(self.attributes)):
                self.individual_size += len(np.unique(self.data[:, a], axis=0))
            self.individual_size += self.target_bits

            self.member_indices = [len(val) for key, val in self.attribute_vals.items()]
        
        self.member_indices = np.cumsum(self.member_indices)
        
        
    def print_data(self):
        print("################## Print Start ##################")
        print("Attributes:", self.attributes)
        print("Target:", self.target_name)
        print("Classes:", self.classes)
        print("Data size:", self.sample_size)
        print("Attribute Values", self.attribute_vals)
        # print("Data:", self.data)
        print("Individual Length:", self.individual_size)
        print("Target encoding: ", self.target_encode_inv)
        
        print("################## Print End ##################\n")
        
        
    def train(self):
        population = self.genPopulation()
        fitness, sample_accuracy = self.getFitness(population)
        
        best_fitness = np.max(fitness)
        best_accuracy = np.max(sample_accuracy)
        self.best_hypothesis = population[np.argmax(fitness)]
        
        # For logging        
        gen_iter = 0
        train_acc_best_log = []
        train_acc_curr_log = []
        test_acc_best_log = []
        test_acc_curr_log = []
        start_time = time.time()
        
        train_acc_best_log.append(np.max(sample_accuracy))
        train_acc_curr_log.append(np.max(sample_accuracy))
        test_acc_best = self.accuracy(self.testX, self.testy)
        test_acc_curr = self.accuracy(self.testX, self.testy)
        
        test_acc_best_log.append(test_acc_best)
        test_acc_curr_log.append(test_acc_curr)
        
        while np.max(fitness) < self.fitness_threshold:
            if self.generation_stop and gen_iter == self.generation_stop:
                break

            ############## Step 1: Select ##############
            pr_h = [1/self.p]*self.p   # Selection probability of each member (initially uniform)
            Ps = []     # New population
            
            # Fitness-proportional Selection
            if self.selection in "Roulette":
                if np.sum(fitness) != 0:
                    for mid in range(self.p):
                        pr_h[mid] = fitness[mid] / np.sum(fitness)
                for _ in range(self.keep_size):
                    Ps.append(np.random.choice(population, p=pr_h))

            # Tournament Selection
            elif self.selection in "Tournament":
                for _ in range(self.keep_size):
                    selected = self.tournament_selection(population, fitness, k=3)
                    Ps.append(selected)
            
            # Rank Selection
            elif self.selection in "Rank":
                # Get sorted indices based on fitness (ascending)
                sorted_fitness = np.argsort(fitness)
                
                # Rank probabilities
                rank_prob = (sorted_fitness + 1) / np.sum(sorted_fitness + 1)
                
                # Probabilitically choose members based on their rank probability
                for _ in range(self.keep_size):
                    Ps.append(np.random.choice(population, p=rank_prob))
            
            
            ############# Step 2: Crossover [Single-point] #############
            crossover_members = []    
            while len(crossover_members) < int(self.replacement_size):
                p1 = np.random.choice(population, p=pr_h) # first parent
                p2 = np.random.choice(population, p=pr_h) # second parent
                p1_sub_hypothesis = list(map(''.join, zip(*[iter(p1)]*self.individual_size)))
                p2_sub_hypothesis = list(map(''.join, zip(*[iter(p2)]*self.individual_size)))

                if self.minmax:
                    d1_p1 = np.random.choice(self.member_indices[:-1])
                    d2_p1 = np.random.choice(self.member_indices[self.member_indices > d1_p1])
                else:
                    d1_p1 = np.random.randint(1, len(p1_sub_hypothesis[0]))
                    d2_p1 = np.random.randint(d1_p1, len(p1_sub_hypothesis[0]))
                
                sub_h_idx1_p1 = np.random.randint(len(p1_sub_hypothesis))
                sub_h_idx2_p1 = np.random.randint(sub_h_idx1_p1, len(p1_sub_hypothesis))
                
                bound1 = p1[self.individual_size*sub_h_idx1_p1 + d1_p1 : self.individual_size*sub_h_idx2_p1 + d2_p1]
            
                sub_h_idx1_p2 = np.random.randint(len(p2_sub_hypothesis))
                sub_h_idx2_p2 = np.random.randint(sub_h_idx1_p2, len(p2_sub_hypothesis))
                bound2 = p2[self.individual_size*sub_h_idx1_p2 + d1_p1 : self.individual_size*sub_h_idx2_p2 + d2_p1]
                
                c1 = p1[:self.individual_size*sub_h_idx1_p1 + d1_p1] + bound2 + p1[self.individual_size*sub_h_idx2_p1 + d2_p1:]
                c2 = p2[:self.individual_size*sub_h_idx1_p2 + d1_p1] + bound1 + p2[self.individual_size*sub_h_idx2_p2 + d2_p1:]
                
                if self.minmax:
                    if self.isGoodMember_minmax(c1) and self.isGoodMember_minmax(c2):
                        crossover_members.append(c1)
                        crossover_members.append(c2)
                else:
                    if self.isGoodMember(c1) and self.isGoodMember(c2):
                        crossover_members.append(c1)
                        crossover_members.append(c2)
                
            Ps += crossover_members # add childrens to population
            
                        
            ############# Step 3: Mutate #############
        
            mutation_len = math.ceil(self.m * self.p)
            mutated_members = []

            while len(mutated_members) < mutation_len:
                member_id = np.random.choice(len(Ps))
                bit_loc = np.random.choice(len(Ps[member_id]))
                curr_bit = Ps[member_id][bit_loc]                
                new_bit = "0" if int(curr_bit) else "1"
                mutated = Ps[member_id][:bit_loc] + new_bit + Ps[member_id][bit_loc+1:]
                
                if self.minmax:
                    if self.isGoodMember_minmax(mutated):
                        Ps.pop(member_id)
                        mutated_members.append(mutated)
                else:
                    if self.isGoodMember(mutated):
                        Ps.pop(member_id)
                        mutated_members.append(mutated)
            
            ############# Step 4: Update #############

            Ps += mutated_members
            population = np.array(Ps)
        
            ############# Step 5: Evaluate #############
            
            fitness, sample_accuracy = self.getFitness(Ps)
            
            if np.max(fitness) > best_fitness:
                best_fitness = np.max(fitness)
                best_accuracy = np.max(sample_accuracy)
                self.best_hypothesis = Ps[np.argmax(fitness)]
                
            test_acc_best = self.accuracy(self.testX, self.testy)
            test_acc_curr = self.accuracy(self.testX, self.testy, Ps[np.argmax(fitness)])
            
            print(f"Generation: {gen_iter} | Best Fitness: {round(best_fitness,3)} | Running Fitness: {round(np.max(fitness),3)} | Train Accuracy: {best_accuracy} | Test Accuracy: {test_acc_best}")
            # print("Best Hypothesis: ", list(map(''.join, zip(*[iter(self.best_hypothesis)]*self.individual_size))))
            
            gen_iter += 1
            train_acc_best_log.append(best_accuracy)
            train_acc_curr_log.append(np.max(sample_accuracy))
            test_acc_best_log.append(test_acc_best)
            test_acc_curr_log.append(test_acc_curr)
            

            if (gen_iter%50 == 0) and self.plotOn:
                import matplotlib.pyplot as plt
                plt.plot(train_acc_curr_log, label="Train Accuracy")
                plt.plot(test_acc_curr_log, label="Test Accuracy")
                plt.title("Generation vs Accuracy [Current best performing hypothesis]")
                plt.xlabel("Generation")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid()
                plt.show()
                
                plt.plot(train_acc_best_log, label="Train Accuracy")
                plt.plot(test_acc_best_log, label="Test Accuracy")
                plt.title("Generation vs Accuracy [Overall best performing hypothesis]")
                plt.xlabel("Generation")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid()
                plt.show()
        print(f"Total time: {time.time() - start_time} seconds")
                    
        self.showLearnedRules()
        return train_acc_best_log, test_acc_best_log
    
    def tournament_selection(self, population, fitness, k):
        # radomly select k member indices
        idxs = np.random.choice(len(population), size=k, replace=False)
        
        # get the member idx corresponding to best fitness
        return population[idxs[np.argmax(fitness[idxs])]]
        
    
    def isGoodMember(self, member):
        sub_hypothesis = list(map(''.join, zip(*[iter(member)]*self.individual_size)))
        
        # Cap the rule set in one hypothesis to 50% of train data (e.g. use max 5 rules for 10 samples)
        if len(sub_hypothesis) > 3:
            return False            
        
        for sub_h in sub_hypothesis:
            sub_h_arr = np.array(list(sub_h))
            
            if np.all(sub_h_arr[:-1] == "1"):
                return False
            
            sub_h_decomposed = np.split(sub_h_arr, self.member_indices)
            
            # Check for attribute with all zeros
            for i in range(len(sub_h_decomposed)-1): # skip target
                if "1" not in sub_h_decomposed[i]:
                    return False
        return True
    
    # Minmax checker
    def isGoodMember_minmax(self, member):
        sub_hypothesis = list(map(''.join, zip(*[iter(member)]*self.individual_size)))
        
        # Cap the rule set in one hypothesis to 50% of train data (e.g. use max 5 rules for 10 samples)
        if len(sub_hypothesis) > self.max_allowed_rules:
            return False 
        
        for sub_h in sub_hypothesis:
            # if target is invalid (not present in encoding)
            if sub_h[-self.target_bits:] not in list(self.target_encode_inv.keys()):                
                return False
            
            sub_h_arr = np.array(list(sub_h))
            sub_h_decomposed = np.split(sub_h_arr, self.member_indices)[:-1]
            
            if np.all(sub_h_arr[:-1] == "1"):
                return False
        
            for i in range(0, len(sub_h_decomposed)-1, 2):
                min = int(''.join(sub_h_decomposed[i]), 2)
                max = int(''.join(sub_h_decomposed[i+1]), 2)
                if min > max:
                    return False
                
        return True
        

    def genPopulation(self):
        population = []
        
        if not self.minmax:
            while len(population) < self.p:
                member = []
                for _ in range(self.variable_len_h):
                    for _ in range(self.individual_size):
                        member.append(random.choice("01"))
                
                member = "".join(member)
                if self.isGoodMember(member):
                    population.append(member)
        else:
            while len(population) < self.p:
                member = []
                for _ in range(self.variable_len_h):
                    for _ in range(self.individual_size):
                        member.append(random.choice("01"))
                
                member = "".join(member)
                if self.isGoodMember_minmax(member):
                    population.append(member)
                    
        return np.array(population)
    
    
    def getFitness(self, population):
        fitness = []
        accuracy = []
        
        for p in population:
            matches = 0
            sub_hypothesis = list(map(''.join, zip(*[iter(p)]*self.individual_size)))
            
            for s in range(self.sample_size):
                sample = self.data[s]
                sample_target = self.target[s]
                
                # Format the target to str for easy comparison with member
                sample_target = self.target_encode[sample_target]
                
                # Check each sample for all sub_hypothesis
                for sub_h in sub_hypothesis:
                    match = False
                    
                    if self.minmax:
                        sub_h_arr = np.array(list(sub_h))
                        sub_h_decomposed = np.split(sub_h_arr, self.member_indices)[:-1]
                        
                        match_count = 0
                        for i in range(len(self.attributes)):
                            min_key = ''.join(sub_h_decomposed[2*i])
                            max_key = ''.join(sub_h_decomposed[2*i + 1])
                            
                            if min_key in list(self.encodings[i].keys()) and max_key in list(self.encodings[i].keys()):
                                min = int(self.encodings[i][min_key])
                                max = int(self.encodings[i][max_key])

                                if sample[i] > min and sample[i] <= max:
                                    match_count += 1
                                    
                        if match_count == len(self.attributes):
                            if (sample_target == sub_h[-self.target_bits:]):
                                match = True
                    else:
                        
                        sample_map = ""
                        
                        for i in range(len(self.attributes)):
                            sample_sub_map = (self.attribute_vals[self.attributes[i]] == sample[i]).astype(int)
                            sample_map += "".join(map(str,sample_sub_map))
                        
                        sample_map = int(sample_map, base=2)
                        match = (sample_map & int(sub_h[:-self.target_bits], base=2)) == sample_map
                                        
                    if not match:
                        continue
                    
                    # or sample and target both fits -> hypopthesis perfectly represents sample
                    if match and (sample_target == sub_h[-self.target_bits:]):
                        matches += 1
                        break

            fitness.append((matches/self.sample_size)**2)
            accuracy.append(matches/self.sample_size)

        return np.array(fitness), np.array(accuracy)
            
    
    def accuracy(self, test_X, test_y, hypothesis=None):
        if hypothesis:
            sub_hypothesis = list(map(''.join, zip(*[iter(hypothesis)]*self.individual_size)))
        else:
            sub_hypothesis = list(map(''.join, zip(*[iter(self.best_hypothesis)]*self.individual_size)))
            
        test_size = test_X.shape[0]
       
        matches = 0
        accuracy = 0
        
        for s in range(test_size):
            sample = test_X[s]
            sample_target = test_y[s]
                
            # Format the target to str for easy comparison with member
            sample_target = self.target_encode[sample_target]

            # Check each sample for all sub_hypothesis
            for sub_h in sub_hypothesis:
                match = False
                
                if self.minmax:
                    sub_h_arr = np.array(list(sub_h))
                    sub_h_decomposed = np.split(sub_h_arr, self.member_indices)[:-1]
                    
                    match_count = 0
                    for i in range(len(self.attributes)):
                        min_key = ''.join(sub_h_decomposed[2*i])
                        max_key = ''.join(sub_h_decomposed[2*i + 1])
                        
                        if min_key in list(self.encodings[i].keys()) and max_key in list(self.encodings[i].keys()):
                            min = int(self.encodings[i][min_key])
                            max = int(self.encodings[i][max_key])

                            if sample[i] > min and sample[i] <= max:
                                match_count += 1
                                
                    if match_count == len(self.attributes):
                        if (sample_target == sub_h[-self.target_bits:]):
                            match = True
                else:
                    
                    sample_map = ""
                    
                    for i in range(len(self.attributes)):
                        sample_sub_map = (self.attribute_vals[self.attributes[i]] == sample[i]).astype(int)
                        sample_map += "".join(map(str,sample_sub_map))
                    
                    sample_map = int(sample_map, base=2)
                    match = (sample_map & int(sub_h[:-self.target_bits], base=2)) == sample_map
                                    
                if not match:
                    continue
                
                # or sample and target both fits -> hypopthesis perfectly represents sample
                if match and (sample_target == sub_h[-self.target_bits:]):
                    matches += 1
                    break
            
        accuracy = matches/test_size
        return accuracy
    
            
    def generateEncodings(self, iris_train_X):
        iris_train_range = []
        unique_sorted_attr = []
        for i in range(iris_train_X.shape[1]):
            unique_sorted_attr.append(np.sort(np.int32(np.unique(iris_train_X[:, i]))))
            iris_train_range.append(np.int32(np.unique(iris_train_X[:, i])).shape[0])

        print("Attribute range: ", iris_train_range)

        # closest power of 2
        bits_needed = []
        for i in range(iris_train_X.shape[1]):
            for j in range(10):
                if 2**j >= iris_train_range[i]:
                    bits_needed.append(j)
                    break
                
        print("# of bits needed for each attr: ", bits_needed)

        train_encodings = []
        for i in range(iris_train_X.shape[1]):
            n = len(unique_sorted_attr[i])
            all_combos = list(map(list, itertools.product([0, 1], repeat=bits_needed[i])))[:n]
            all_combos = [''.join(map(str, i)) for i in all_combos]
            train_encodings.append(dict(zip(all_combos, unique_sorted_attr[i])))
        return train_encodings, np.array(bits_needed)

    def showLearnedRules(self):
        sub_hypothesis = list(map(''.join, zip(*[iter(self.best_hypothesis)]*self.individual_size)))
        
        print("\n##############################")
        print(f"\nBest Hypothesis contains following {len(sub_hypothesis)} disjuntive rules:")
        # Check each sample for all sub_hypothesis
        for i, sub_h in enumerate(sub_hypothesis):
            print(f"Learned Rule {i+1}")
            rule_str = "If"
            
            sub_h_arr = np.array(list(sub_h))
            sub_h_decomposed = np.split(sub_h_arr, self.member_indices)[:-1]
                
            max_i = len(self.attributes) - 1
            
            for i in range(len(self.attributes)):
                if self.minmax:
                    rule_str += " ( "
                    
                    # Rules string for minmax (iris)
                    min_key = ''.join(sub_h_decomposed[2*i])
                    max_key = ''.join(sub_h_decomposed[2*i + 1])
                    
                    if min_key in list(self.encodings[i].keys()) and max_key in list(self.encodings[i].keys()):
                        min = int(self.encodings[i][min_key])
                        max = int(self.encodings[i][max_key])
                    else:
                        min = int(list(self.encodings[i].keys())[0]) + int(min_key, 2)
                        min = int(list(self.encodings[i].keys())[0]) + int(max_key, 2)
                        
                    rule_str += str(min/10) + " < " + self.attributes[i] + " <= " + str(max/10)
                    rule_str += " ) "
                    
                else:
                    # Rules string for tennis
                    rule_str += " ( " + self.attributes[i] + " is "
                    or_str_count = 0
                    for val in range(len(sub_h_decomposed[i])):
                        if sub_h_decomposed[i][val] == "1":
                            rule_str += self.attribute_vals[self.attributes[i]][val]

                            if or_str_count < (sum(list(map(int, sub_h_decomposed[i]))) - 1):
                                rule_str += " or "
                                or_str_count += 1
                    rule_str += " ) "
                    
                if i < max_i:
                    rule_str += "AND"
                
            rule_str += " -> THEN target = " + self.target_encode_inv[sub_h[-self.target_bits:]]
            print(rule_str)
            print()
