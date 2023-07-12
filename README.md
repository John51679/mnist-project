# mnist-project
A given University task to identify handwritten digits, using a neural network and fine-tuning it and then explore efficient solutions, by cutting as many inputs as possible while preserving accuracy, with the use of a genetic algorithm.

The project was implemented in Python.

# The mnist dataset

### You can find the mnist dataset __[here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)__

## Neural Network

For the neural network part, with the help of keras and tensorflow libraries, we create the architecture of the neural network and later we fine-tune its hyperparameters with the use of k-fold cross validation that sklearn library provides through GridSearchCV class.

## Genetic Algorithm

After the neural network is built and fine tuned, we then perform an input reduction optimization using a genetic algorithm. In this part we attempt to reduce the number of inputs that the aforementioned neural network has, while at the same time preserving its accuracy. That approach not only does reduce the time complexity, but at the same time it shows how the neural network is capable of handling input noise.
