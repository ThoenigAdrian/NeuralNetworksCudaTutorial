# Part 5 - Backpropagation

Last time we finished the feedforward code it now supports multiple inputs, and the neural network can have multiple layers. Now we will implement backpropagation so we can get the gradients and finally do gradient descent.

## 1. Motivation

After we feedforward our input through the network we get some output activations.
This output activations however are rather random since our neural network isn’t trained yet.
So we need to train it.  For training we can use gradient descent. 
But in order to use gradient descent we need to compute the gradients first. And the way compute the gradients is through backpropagation.

## 2. Backpropagation Math

### 2.1 Initial Gradients

*Single Neuron*

We start by looking at the last layer first and then how to move backwards through the network. So first we need an initial gradient for the output neuron usually this comes from a loss function.
For starters we are just going to assume that the gradients of the output neurons can be described as “target – activations”. Where target is the value, we want our neural network to produce. 
For example if let’s say we want our neuron to produce the value 0.8 but it actually produces 0.67.
Then target – activations will be 0.13 . So the gradient for this neuron would be 0.13

<img width="940" height="203" alt="image" src="https://github.com/user-attachments/assets/2f5f3668-b788-429a-84cd-a6a0d97486ed" />

Each target has its corresponding input.
In the current image we can see only one output neuron.

*Multiple Neurons*

Each neuron has its own target and it’s own activation.
We compute all the gradients for all the biases.
For every neuron we use it desired target and it’s current activation value.
