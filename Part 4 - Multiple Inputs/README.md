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

In this example image we would have `3` inputs and `6` hidden neurons therefore resulting in `18` cuda threads. 
`nr_threads = nr_inputs * nr_hidden_neurons` (side note: of course the number of hidden neurons is different for every layer in our code we are just setting the number of hidden neurons to the max of all layers , to make sure we have enough threads. )

### Details

Alright let’s have a closer look at what each individual thread will do. We are going to use the following example to visualize everything. ![SetupExampleThreadingStrat](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/d19894a6-10e8-4d47-9840-980df95713e9)

We use a simple neural network with 4 input neurons and 3 output neurons. The inpput matrix will consist of 3 inputs. This results in  `9` total threads. `3 (nr_inputs) * 3( nr_neurons) = 9 (total threads)`   Now let’s have a look at what each individual thread does.

![detailedthreadingstrat](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/63d2a027-c9ca-4de3-8967-59ed3ac1e10b)

**Thread 0** (threadIdx.x=0, threadIdx.y=0) will take the first input and compute the activations for the first output neuron. 

**Thread 1** threadIdx.x=1, threadIdx.y=0) will also take the first input but will take care of the second output neuron. 

**Thread 2** tthreadIdx.x=2, threadIdx.y=0)akes care of the final output neuron.

Next we move on to thread 3 – 5 . This 3 threads will take care of the 2nd input. 

**Thread 3** (threadIdx.x=0, threadIdx.y=1) will take the `2nd` input and compute the activations for the first output neuron

**Thread 4** (threadIdx.x=1, threadIdx.y=1) also takes the `2nd` input but takes care of the second ouput neuron.

I assume you can see the pattern here ;) . 

## Data Structures


Now that we want to handle multiple inputs we need to change our Memory Structure a little bit.
The structure of the weights and biases stay the same. We need to change the activations array and the z values array to accommodate multiple inputs.
Here are the changes we are going to make to the data structure.

CLICK IMAGE TO ENLARGE
![datastructures](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/1b9235ad-5cba-4dca-a081-755a0928e056)



The order of the layers in the memory stays the same. But each layer section now holds multiple inputs instead of just one. 

## Main

### Kernel Call
There are 2 changes we need to do to the code in the main function. 
1.	We need to change how we call the kernel
2.	We need to change the data structures for activations, and z values to be compatible with multiple inputs.

So, for calling the kernel the main thing which changes is that now we need more threads.

We need one thread per input and neuron. So, if we have 3 Inputs and 6 Neurons we need 18 Threads. We can specify this via a 2-dimensional tuple. Which specifies the dimensions of the threadBlock we are going to use.
The number of threads in the x dimension is going to be equal to the number of neurons the largest layer has:

So if we have a look at the shape of our neural network:
![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/29a9d90a-db66-4527-b382-cb3c584edc9f)

8,6,4,1
We will need 6 threads because we have at most 6 neurons. 
So the thread block dimensions in the x-axis will be 6.

The number of threads in the y dimension is going to be equal to the number of inputs.

![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/d24eac79-e749-48e2-8fd7-5275915dc564)

That’s it for the Kernal Call now let’s move on to the Data Structures.

### Data Structures

Now let’s have a look at a side by side comparison between the	code of the previous video and our current code. First we introduce a variable which defines the number of inputs. 

![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/ee814de5-835e-43eb-a3be-e4332711c3a3)

```
int main()
{
	const int shape_length = 4;
	int shape[shape_length] = { 8, 6, 4, 1 };
	const int NR_INPUTS = 3;

	// Initialize weights on CPU/RAM
	int nr_weights = 0;

	for (int shape_index = 0; shape_index < shape_length - 1; shape_index++)
	{
		nr_weights += shape[shape_index] * shape[shape_index + 1];
	}

	float *host_weights = new float [nr_weights] {1.62f, -0.61f, -0.53f, -1.07f, 0.87f, -2.30f, 1.74f, -0.76f, 0.32f, -0.25f, 1.46f, -2.06f, -0.32f, -0.38f, 1.13f, -1.10f, -0.17f, -0.88f, 0.04f, 0.58f, -1.10f, 1.14f, 0.90f, 0.50f, 0.90f, -0.68f, -0.12f, -0.94f, -0.27f, 0.53f, -0.69f, -0.40f, -0.69f, -0.85f, -0.67f, -0.01f, -1.12f, 0.23f, 1.66f, 0.74f, -0.19f, -0.89f, -0.75f, 1.69f, 0.05f, -0.64f, 0.19f, 2.10f, 0.12f, 0.62f, 0.30f, -0.35f, -1.14f, -0.35f, -0.21f, 0.59f, 0.84f, 0.93f, 0.29f, 0.89f, -0.75f, 1.25f, 0.51f, -0.30f, 0.49f, -0.08f, 1.13f, 1.52f, 2.19f, -1.40f, -1.44f, -0.50f, 0.16f, 0.88f, 0.32f, -2.02f};

	// Initialize biases on CPU/RAM
	int nr_neurons = 0;
	int nr_biases = 0;

	for (int shape_index = 0; shape_index < shape_length; shape_index++)
	{
		nr_neurons += shape[shape_index];
	}

	nr_biases = nr_neurons - shape[0];
	float *host_biases = new float [nr_biases] {-0.31f, 0.83f, 0.23f, 0.76f, -0.22f, -0.20f, 0.19f, 0.41f, 0.20f, 0.12f, -0.67f};
```


Next we need to change the size of the activations array previously the number of activations was the same as the number of neurons but now we need one set of activations for every input. So we need to multiply the number of neurons with the number of inputs.

![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/e7821191-2ff5-4661-b040-69c4268f4f25)

Another thing we do for the activations array is to initialize it with enough values for 3 inputs so that's why the list is longer than in the previous code. 
Let's move on to the z values array. Here we encounter a similar situation previously the number of z values was equivalent to the number biases but now we compute a bunch of z values for every intput. So we also need to multiply with the number of inputs here.
