# mnist-project
This project was created as part of "Computational Intelligence" subject in Computer Engineering & Informatics Department (CEID) of University of Patras.The task was to identify handwritten digits, using a neural network, fine-tuning it and then explore efficient solutions, by cutting as many inputs as possible while preserving accuracy, with the use of a genetic algorithm.

The project was implemented in Python.

# The mnist dataset

### You can find the mnist dataset __[here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)__

## Neural Network

For the neural network part, with the help of keras and tensorflow libraries, we create the architecture of the neural network and later we fine-tune its hyperparameters with the use of k-fold cross validation that sklearn library provides through GridSearchCV class.

## Genetic Algorithm

After the neural network is built and fine tuned, we then perform an input reduction optimization using a genetic algorithm. In this part we attempt to reduce the number of inputs that the aforementioned neural network has, while at the same time preserving its accuracy. That approach not only does reduce the time complexity, but at the same time it shows how the neural network is capable of handling input noise.

For the GA's architecture we use:

* For the encoding (Chromosomes of an individual in our Genetic Algorithm
  * A bit array of 784 length, that each index represents the corresponding input index of the neural network. If it's 0 we do not consider this input in the neural net (we make its corresponding weights 0) and if it's 1 we consider it as a valid input. Randomly initialized. However through the 'population_initializer' value, the user can define regions within the chromosome where the values will be initialized as zero.
* For the fitness function we choose the equation
$$f(𝑥, 𝑦) = \frac{𝑐(1 − 𝑥)}{y}$$
where
  * x is the loss of the neural network calculated by the categorical cross entropy loss function
  * y is the number of inputs that the neural network was given
  * c is a constant value which represents the number of the original neural network inputs (784 in this case)
* For the GA selection operator
  * Use of Rank Based Roulette Wheel Selection algorithm
* For the GA crossover operator
  * Use of Uniform crossover function
* For the GA mutation operator
  * Use of elitism method

## Example of the end result

The mnist dataset has 784 input pixels (28x28 size) and 1 output label that contains values ranging from 0 to 9. In this example we explore an instance of this project's output.

![Output](https://github.com/John51679/mnist-project/blob/main/GeneticAlgorithm/Experiments/Final_experiments_TRAINED_rep.jpg)

* Giving all 784 inputs represents our original neural network
* Without retraining indicates that we did not retrain our neural network after the genetic algorithm's output
* With retraining indicates that we did retrain our neural network after the genetic algorithm's output

___
Lastly, for the retraining process we present an instance of a number seven given as input. We show its original input and then the input pixels that genetic algorithm kept.

| Original                  | After Genetic Algorithm                   |
| ------------------------- | ------------------------- |
| ![Server_Original](https://github.com/John51679/mnist-project/blob/main/GeneticAlgorithm/seven.jpg)   | ![Server_Original](https://github.com/John51679/mnist-project/blob/main/GeneticAlgorithm/seven_after.jpg)    |
