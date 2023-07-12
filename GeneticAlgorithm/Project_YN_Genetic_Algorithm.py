import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

class my_Gen:
    def __init__(self,ANN_model,input_dataset,output_dataset,crossover_probability,mutation_probability,chromosomes, population_initializer=None): #Initializations take place here (constructor)
        self.model = ANN_model #Load the saved model
        self.inData = input_dataset #Store the input dataset
        self.outData = output_dataset #Store the output dataset
        self.cols = len(self.inData[0,:]) #Store the number of features (inputs)
        self.population = np.random.randint(0,2,[chromosomes,self.cols]) #Initialize population
        self.fitnesses = [] #Array that will store the fitness function value of each chromosome
        self.Pc = crossover_probability
        self.Pm = mutation_probability
        self.history = [] #History is used to plot the evolution curve
        for i in range(len(self.population[:,0])):
            self.fitnesses.append(0)

            #The code below is defensive programming
        if (self.Pc > 1 or self.Pc < 0):
            err = os.error("Crossover Probability must be between 0 and 1, but a value of "+ str(self.Pc) + " was given ")
            raise ValueError(err)
        if (self.Pm > 1 or self.Pm < 0):
            err = os.error("Mutation Probability must be between 0 and 1, but a value of "+ str(self.Pm) + " was given ")
            raise ValueError(err)
        if population_initializer != None:
            try:
                for i in population_initializer:
                    if (type(i).__name__ == 'int'):
                        self.population[:,i] = 0
                    elif (type(i).__name__ == 'list' or type(i).__name__ == 'range'):
                        for j in i:
                            self.population[:,j] = 0
                    else:
                        err = os.error("Population initializer must be an integer or a list of integers but " + type(i).__name__ + " was given instead in index " + str(population_initializer.index(i)) +"!")
                        raise TypeError()
            except TypeError:
                if (type(population_initializer).__name__ == 'int'):
                    self.population[:,population_initializer] = 0
                else:
                    raise TypeError(err)


    def __fitness_function__(self,loss,input_num): #Calculation of the fitness function
        """Numerator = (1 - loss)
        Denominator = (input_num/self.cols)
        return Numerator/Denominator"""
        return (1 - input_num/self.cols)*(loss + 1)/2

    def fitness_eval(self): #Model evaluation through fitness function
        weights = self.model.weights
        initial_weights = weights[0].numpy()
        for i in range(len(self.population[:,0])):
            temp = np.copy(initial_weights)
            for j in range(len(self.population[0,:])):
                if self.population[i,j] == 0:
                    temp[j,:] = 0
            weights[0].assign(temp)
            loss, _ = self.model.evaluate(self.inData,self.outData,verbose=False)
            input_num = self.population[i,:].tolist().count(1)
            self.fitnesses[i] = self.__fitness_function__(loss,input_num)
            weights[0].assign(initial_weights)
        self.history.append(max(self.fitnesses))

    def rank_based_roulette_wheel_selection(self): #Base process of rank based roulette wheel selection as described in the project report
        Q = []
        rank = 1
        ranks = np.zeros([1,len(self.population[:,0])])
        fit = self.fitnesses.copy()
        selected_pop = np.zeros([len(self.population[:,0]),len(self.population[0,:])])
        tmp = min(fit)
        for i in range(len(self.population[:])):
            min_index = fit.index(min(fit))
            if (tmp == fit[min_index]): ranks[0,min_index] = rank
            else:
                rank +=1
                ranks[0,min_index] = rank
            tmp = min(fit)
            fit[min_index] = np.inf
        sums = np.sum(ranks)
        Ps = ranks / sums
        for i in range(len(self.population[:,0])):
            Q.append(sum(Ps[0,:i + 1]))
        for i in range(len(self.population[:,0])):
            r = random.random()
            index = 0
            while(r > Q[index]):
                index += 1
            selected_pop[i,:] = self.population[index,:]
        return selected_pop

    def __swap__(self,a,b,temp,index): #swap function for the crossover process
        temp[0,index] = b
        temp[1,index] = a
        return temp

    def crossover_process(self, selected_pop): #Base function for the crossover process
        crossover_array = np.zeros([1,self.cols + 1])
        for i in range(len(selected_pop[:,0])):
            r = random.random()
            if (r < self.Pc):
                crossover_array = np.insert(crossover_array,len(crossover_array),np.insert(selected_pop[i,:],len(selected_pop[i,:]),i),axis=0)
        crossover_array = np.delete(crossover_array,0,axis=0)
        size = len(crossover_array[:,0])
        if (size%2 == 1):
            size -= 1
            r = random.randint(0,size)
            temp = np.array(crossover_array[size,:])
            temp = np.reshape(temp,[1,len(temp)])
            temp = np.insert(temp,1,crossover_array[r,:],axis=0)
            temp = np.insert(temp,2,np.random.randint(0,2,[1,len(temp[0,:])]),axis=0)
            for i in range(len(temp[0,:])-1):
                if temp[2,i] == 1: temp = self.__swap__(temp[0,i],temp[1,i],temp,i)
            selected_pop[int(temp[0,len(temp[0,:])-1]),:] = temp[0,:len(temp[0,:]) - 1]
            selected_pop[int(temp[1,len(temp[1,:])-1]),:] = temp[1,:len(temp[1,:]) - 1]
        for i in range(0,int(size/2),2):
            temp = np.array(crossover_array[i, :])
            temp = np.reshape(temp, [1, len(temp)])
            temp = np.insert(temp, 1, crossover_array[i+1, :], axis=0)
            temp = np.insert(temp, 2, np.random.randint(0, 2, [1, len(temp[0, :])]), axis=0)
            for j in range(len(temp[0, :])-1):
                if temp[2, j] == 1: temp = self.__swap__(temp[0, j], temp[1, j], temp, j)
            selected_pop[int(temp[0, len(temp[0, :]) - 1]), :] = temp[0, :len(temp[0, :]) - 1]
            selected_pop[int(temp[1, len(temp[1, :]) - 1]), :] = temp[1, :len(temp[1, :]) - 1]
        return selected_pop

    def mutation_process(self,selected_pop): #Base function for the mutation process
        best_chrom_index = self.fitnesses.index(max(self.fitnesses))
        for i in range(len(selected_pop[:,0])):
            if(self.fitnesses[i] != self.fitnesses[best_chrom_index]):
                r = np.random.random([1,self.cols])
                for j in range(self.cols):
                    if (r[0,j] < self.Pm):
                        if selected_pop[i,j] == 1: selected_pop[i,j] = 0
                        else: selected_pop[i,j] = 1
        return selected_pop

    def run(self,plot = False): #This function will be used to run everything the genetic algorithm needs
        counter = 0
        stop = 0
        self.fitness_eval()
        tmp = 0
        while(counter != 1000):
            temp = self.fitnesses.copy()
            max_index = temp.index(max(temp))
            sel_pop = self.rank_based_roulette_wheel_selection()
            sel_pop = self.crossover_process(sel_pop)
            self.population = np.copy(sel_pop)
            self.fitness_eval()
            if (self.Pm != 0):
                sel_pop = self.mutation_process(sel_pop)
                self.population = np.copy(sel_pop)
                self.fitness_eval()
            fit_mean = sum(self.fitnesses)/len(self.fitnesses)
            #print(temp[max_index], fit_mean)
            if (temp[max_index] < fit_mean + temp[max_index]*0.015): break
            if (tmp >= temp[max_index]): stop += 1
            else:
                stop = 0
                tmp = temp[max_index]
            if (stop == 4): break
            counter += 1
        max_index = self.fitnesses.index(max(self.fitnesses))
        input_num = self.population[max_index, :].tolist().count(1)
        print("Best solution was found with number of inputs {0} and Accuracy {1}".format(input_num, self.history[-1]))

        hist = self.best_solution_history(plot)
        best = self.population[max_index,:]
        return best

    def best_solution_history(self,plot=False):
        if plot:
            plt.plot(self.history)
            plt.title('My model')
            plt.ylabel('Accuracy')
            plt.xlabel('Generations')
            plt.show()
        return self.history


def scale(inData):
    inData = np.float64(inData)
    for i in range(len(inData[0,:])):
        if inData[:, i].std() != 0:
            inData[:, i] = np.divide(np.subtract(inData[:, i], inData[:, i].mean()), inData[:, i].std())
        else:
            inData[:, i] = np.subtract(inData[:, i], inData[:, i].mean())
    return inData

"""Store model from Part A and start preprocessing again"""
json_file = open('../model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("../model.h5")

trainDF = pd.read_csv("../mnist_train.csv")
testDF = pd.read_csv("../mnist_test.csv")

train_input = trainDF.iloc[:,1:].values
train_output = trainDF.iloc[:,0].values
test_input = testDF.iloc[:,1:].values
test_output = testDF.iloc[:,0].values

train_input = scale(train_input)
test_input = scale(test_input)

trainEN = np_utils.to_categorical(train_output).astype(float)
testEN = np_utils.to_categorical(test_output).astype(float)
"""End of preprocessing"""


optimizer = SGD(learning_rate=0.001,momentum=0.6)
loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

a = [range(0,60),range(750,784)] #This is used for the population initializer

"""This was used for the experiments that required 10 runs each"""
"""
best_hist = np.zeros([10,1001])
gens = np.zeros([1,10])
input_num = np.zeros([1,10])
best_chroms = np.zeros([10,784])
accs = 0

for i in range(10):
    g = my_Gen(loaded_model,test_input,testEN,0.9,0.01,4)
    res = g.run()
    hist = g.best_solution_history(False)
    best_chroms[i,:] = res
    best_hist[i,:len(hist)] = hist
    gens[0,i] = len(hist)
    input_num[0,i] = best_chroms[i,:].tolist().count(1)

    weights = loaded_model.weights
    initial_weights = weights[0].numpy()
    temp = np.copy(initial_weights)
    for j in range(len(best_chroms[0, :])):
        if best_chroms[0, j] == 0:
            temp[j, :] = 0
    weights[0].assign(temp)
    _, acc = loaded_model.evaluate(test_input, testEN, verbose=False)
    accs += acc
    weights[0].assign(initial_weights)
    del g

accs = accs / 10
#plt.show()
gens = int(np.mean(gens[0,:]))
#gens = int(gens.max())
input_num = int(np.mean(input_num[0,:]))
history_mean = np.zeros([1,gens])
for i in range(gens):
    nonzeros = best_hist[:,i].nonzero()
    if (len(nonzeros[0]) == 0):
        break
    history_mean[0,i] = best_hist[nonzeros[0].tolist(),i].mean()
plt.plot(history_mean[0,:])
plt.title('Genetic Algorithm (10 runs)')
plt.ylabel('Accuracy')
plt.xlabel('Generations')
plt.show()
print("Mean number of inputs = {0}\nMean number of generations = {1}".format(input_num,gens - 1))
print("Mean accuracy {0}".format(accs))
"""
"""end"""

"""Results with untrained Neural Network"""
res = []
g = my_Gen(loaded_model,test_input,testEN,0.6,0,200,a)
result = g.run(plot = True)
_, acc = loaded_model.evaluate(test_input, testEN, verbose=False)
res.append(acc)
weights = loaded_model.weights
initial_weights = weights[0].numpy()
temp = np.copy(initial_weights)
for j in range(len(result)):
    if result[j] == 0:
        temp[j, :] = 0
weights[0].assign(temp)
_, acc = loaded_model.evaluate(test_input, testEN, verbose=False)
res.append(acc)
weights[0].assign(initial_weights)
input_num = result.tolist().count(1)
print("Results with 784 inputs -> {0}\nResults with {1} inputs without retraining-> {2}".format(res[0],input_num,res[1]))
"""End"""

"""Results with retrained Neural network"""
def model():
    model = Sequential()
    model.add(Dense(397, input_dim=input_num, activation='relu'))
    model.add(Dense(198,activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    optimizer = SGD(learning_rate=0.001,momentum=0.6)
    """we have two model.compile below. The first uses CE loss function and the other uses MSE"""
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

result = np.reshape(result,[1,-1])
indexes = result[:].nonzero()[1].tolist()

train_input = trainDF.iloc[:,indexes].values
test_input = testDF.iloc[:,indexes].values

estimator = KerasClassifier(build_fn=model, epochs=100, batch_size=100, verbose = 0)
estimator.fit(train_input,trainEN)
acc = estimator.score(test_input,testEN)
res.append(acc)

print("Results with 784 inputs -> {0}\nResults with {1} inputs with retraining -> {2}".format(res[0],input_num,res[2]))

"""end"""