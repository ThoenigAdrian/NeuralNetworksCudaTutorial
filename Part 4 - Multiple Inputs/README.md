# Part 4 - Multiple Inputs
Last time we implemented the feedforward method for multiple layers. However, one limiting factor is that right now we are only using a single input vector. A modern GPU can run thousands of threads at once so we should make use of that opportunity and code our kernel, so it passes multiple inputs through the neural network at once. 

More inputs means we need more threads so let's cover how we handle this.

## Threading Strategy


In this list of threads each thread corresponded to a specific neuron in the current layer of our neural network. 
Now that we want to take care of multiple inputs, we are going to change that. In our new threading strategy, we will have one thread per neuron for every input. So, we are going to use a thread block. 

What does this mean ? 

In CUDA we can have something which is called a Thread Block. It's basically just a bunch of threads like before but we have two different threadIndex variables. threadIdx.x and threadIdx.y.

![threadblockvsthreadlist](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/9ac95e3c-a2c7-4bef-b75a-4b77008ac4a2)

The x-coordinate of the thread (threadIdx.x) will decide for what neuron we will compute the activation and the y-coordinate (threadIdx.y) will tell us which input we are processing.

In this example image we would have 3 inputs and 6 hidden neurons therefore resulting in 18 cuda threads. 
`nr_threads = nr_inputs * nr_hidden_neurons` (side note: of course the number of hidden neurons is different for every layer in our code we are just setting the number of hidden neurons to the max of all layers , to make sure we have enough threads. )
``````
### Details

Alright let’s have a closer look at what each individual thread will do. We are going to use the following example to visualize everything. ![SetupExampleThreadingStrat](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/d19894a6-10e8-4d47-9840-980df95713e9)

We use a simple neural network with 4 input neurons and 3 output neurons. The inpput matrix will consist of 3 inputs. This results in  9 total threads. 3 (nr_inputs) * 3( nr_neurons)   Now let’s have a look at what each individual thread does.
