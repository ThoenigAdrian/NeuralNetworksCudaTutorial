# Part 4 - Multiple Inputs
Last time we implemented the feedforward method for multiple layers. However, one limiting factor is that right now we are only using a single input vector. A modern GPU can run thousands of threads at once so we should make use of that opportunity and code our kernel, so it passes multiple inputs through the neural network at once. 


More inputs means we need more threads so let's cover how we handle this.

So let’s start with that:
## 1. Threading Strategy


Until now we just had a list of threads. 

In this list of threads each thread corresponded to a specific neuron in the current layer of our neural network. 

![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/e20fd65f-809a-4e7a-a8a6-bc42e5c04207)


Now that we want to take care of multiple inputs, we are going to change that. In our new threading strategy, we will have one thread per neuron for every input. So, we are going to use a thread block. 

What does this mean ? 

In CUDA we can have something which is called a Thread Block. It's basically just a bunch of threads like before but we have two different threadIndex variables. threadIdx.x and threadIdx.y.

![threadblockvsthreadlist](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/9ac95e3c-a2c7-4bef-b75a-4b77008ac4a2)

The x-coordinate of the thread (threadIdx.x) will decide for what neuron we will compute the activation and the y-coordinate (threadIdx.y) will tell us which input we are processing.

In this example image we would have `3` inputs and `6` hidden neurons therefore resulting in `18` cuda threads. 
`nr_threads = nr_inputs * nr_hidden_neurons` (side note: of course the number of hidden neurons is different for every layer in our code we are just setting the number of hidden neurons to the max of all layers , to make sure we have enough threads. )

### 1.1 Details

Alright let’s have a closer look at what each individual thread will do. We are going to use the following example to visualize everything. ![SetupExampleThreadingStrat](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/d19894a6-10e8-4d47-9840-980df95713e9)

We use a simple neural network with 4 input neurons and 3 output neurons. The inpput matrix will consist of 3 inputs. This results in  `9` total threads. `3 (nr_inputs) * 3( nr_neurons) = 9 (total threads)`   Now let’s have a look at what each individual thread does.

![detailedthreadingstrat](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/63d2a027-c9ca-4de3-8967-59ed3ac1e10b)

**Thread 0** (threadIdx.x=0, threadIdx.y=0) will take the first input and compute the activations for the first output neuron. 

**Thread 1** (threadIdx.x=1, threadIdx.y=0) will also take the first input but will take care of the second output neuron. 

**Thread 2** (threadIdx.x=2, threadIdx.y=0)akes care of the final output neuron.

Next we move on to thread 3 – 5 . This 3 threads will take care of the 2nd input. 

**Thread 3** (threadIdx.x=0, threadIdx.y=1) will take the `2nd` input and compute the activations for the first output neuron

**Thread 4** (threadIdx.x=1, threadIdx.y=1) also takes the `2nd` input but takes care of the second ouput neuron.

**To summarize y coordinate decides the input nr , x coordinate the neuron nr**

## 2. Data Structures


Now that we want to handle multiple inputs we need to change our Memory Structure a little bit.
We need to change the activations array and the z values array to accommodate multiple inputs.
Here are the changes we are going to make to the data structure.

**CLICK IMAGE TO ENLARGE**
![datastructures](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/1b9235ad-5cba-4dca-a081-755a0928e056)

The order of the layers in the memory stays the same. But each layer section now holds multiple inputs instead of just one. 

So the hierarchy from big to small:

1. Layers
2. Inputs
3. Neurons


## 3. Code Changes

### 3.1 Main - Code

There are 2 changes we need to do to the code in the main function. 
1.	We need to change how we call the kernel
2.	We need to change the data structures for activations, and z values to be compatible with multiple inputs.


#### 3.1.1 Kernel Call - Code


So, for calling the kernel the main thing which changes is that now we need more threads.

We need one thread per input and neuron. So, if we have 3 Inputs and 6 Neurons we need 18 Threads. We can specify this via a 2-dimensional tuple. Which specifies the dimensions of the threadBlock we are going to use.
The number of threads in the x dimension is going to be equal to the number of neurons the largest layer has:

So if we have a look at the shape of our neural network:
![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/29a9d90a-db66-4527-b382-cb3c584edc9f)

`shape = [8,6,4,1]`
`const inst NR_INPUTS = 3;`

We will need 6 threads in the x-dimension because we have at most 6 neurons. 
The number of threads in the y dimension is going to be equal to the number of inputs in this code I just set it to 3.

```c++
// Call cuda kernel
int nr_threads_x_dimension = *std::max_element(shape + 1, shape + shape_length);
dim3 thread_block_dimensions(nr_threads_x_dimension, NR_INPUTS);
multiple_inputs << <1, thread_block_dimensions >> > (d_weights, d_biases, d_z, d_activations, d_shape, shape_length);
```


That’s it for the Kernal Call now let’s move on to the Data Structures.


#### 3.1.2 Data Structures - Code

Now let’s have a look the difference  comparison between the code of the previous video and our current code. First we introduce a variable which defines the number of inputs. _


```diff
int main()
{
	const int shape_length = 4;
	int shape[shape_length] = { 8, 6, 4, 1 };
+	const int NR_INPUTS = 3;

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

Next we need to change the size of the activations array and z_array. 
Previously nr_activations = nr_neurons but now we need **one set of activations for every input**. So we need to multiply the number of neurons with the number of inputs.

```diff
-       float *host_z = new float [nr_biases] {0.0f};
+       int nr_z = nr_biases * NR_INPUTS;
+       float *host_z = new float [nr_z] {0.0f};


        // Calculate the amount of memory needed so we can provide this information to cuda malloc
        const size_t bytes_biases = nr_biases * sizeof(float);
-       const size_t bytes_z = nr_biases * sizeof(float);
+       const size_t bytes_z = nr_z * sizeof(float);
        const size_t bytes_weights = nr_weights * sizeof(float);
-       const size_t bytes_activations = nr_neurons * sizeof(float);
+       const size_t bytes_activations = nr_activations * sizeof(float);
        const size_t bytes_shape = sizeof(int) * shape_length;


        // Allocate GPU device memory
        float *d_biases, *d_weights, *d_activations, *d_z;
        int *d_shape;
        cudaMalloc(&d_biases, bytes_biases);
        cudaMalloc(&d_weights, bytes_weights);
        cudaMalloc(&d_activations, bytes_activations);
        cudaMalloc(&d_z, bytes_z);


        // Initialize z Matrix
-       float *host_z = new float [nr_biases] {0.0f};
+       int nr_z = nr_biases * NR_INPUTS;
+       float *host_z = new float [nr_z] {0.0f};


        // Calculate the amount of memory needed so we can provide this information to cuda malloc
        const size_t bytes_biases = nr_biases * sizeof(float);
-       const size_t bytes_z = nr_biases * sizeof(float);
+       const size_t bytes_z = nr_z * sizeof(float);
        const size_t bytes_weights = nr_weights * sizeof(float);
-       const size_t bytes_activations = nr_neurons * sizeof(float);
+       const size_t bytes_activations = nr_activations * sizeof(float);
        const size_t bytes_shape = sizeof(int) * shape_length;

```
Another thing we do for the activations array is to initialize it with enough values for 3 inputs so that's why the list is longer than in the previous code. 
```c++
	
	// The first 8 values are our inputs rest of the array can be initialized with 0.0
	int nr_activations = nr_neurons * NR_INPUTS;
	float *host_activations = new float [nr_activations] {0.38f, 0.12f, 1.13f, 1.20f, 0.19f, -0.38f, -0.64f, 0.42f, 0.76f, -0.36f, -0.23f, -0.89f, -0.01f, -0.08f, -0.26f, -0.13f, -0.55f, -0.42f, -0.39f, -0.83f, 0.87f, 0.44f, -0.45f, -0.52f};
	
	// Initialize z Matrix
	int nr_z = nr_biases * NR_INPUTS;
	float *host_z = new float [nr_z] {0.0f};
```

Let's move on to the z values array. Here we encounter a similar situation previously the number of z values was equivalent to the number biases but now we compute a bunch of z values for every intput. So we also need to multiply with the number of inputs here.

#### 3.1.3 Printing the result - Code 

The final thing left to do is to adapt the printing to the new data structure so we can see what values our kernel computes.

```diff
        int z_offset = 0;
        for (int shape_index = 1; shape_index < shape_length; shape_index++)
        {
                std::cout << "Z Values " << shape_index << ". hidden layer" << std::endl;
+
                for (int neuron_nr = 0; neuron_nr < shape[shape_index]; neuron_nr++)
                {
-                       std::cout << host_z[neuron_nr + z_offset] << std::endl;
+                       std::cout << "[";
+                       for (int input = 0; input < NR_INPUTS; input++)
+                       {
+                               std::cout << host_z[z_offset + input * shape[shape_index] + neuron_nr] << ", ";
+                       }
+                       std::cout << "]" << std::endl;
+
                }
-               z_offset += shape[shape_index];
+               z_offset += shape[shape_index] * NR_INPUTS;
        }

-       int activations_offset = shape[0]; // Skip input values
+       int activations_offset = shape[0] * NR_INPUTS; // Skip input values
        for (int shape_index = 1; shape_index < shape_length; shape_index++)
        {
                std::cout << "Activations " << shape_index << ". hidden layer" << std::endl;

                for (int neuron_nr = 0; neuron_nr < shape[shape_index]; neuron_nr++)
                {
-                       std::cout << host_activations[neuron_nr + activations_offset] << std::endl;
+                       std::cout << "[";
+                       for (int input = 0; input < NR_INPUTS; input++)
+                       {
+                               std::cout << host_activations[activations_offset + input * shape[shape_index] + neuron_nr] << ", ";
+                       }
+                       std::cout << "]" << std::endl;
                }
-               activations_offset += shape[shape_index];
+               activations_offset += shape[shape_index] * NR_INPUTS;
        }

        getchar();


        return;

```


### 3.2 Kernel Changes - Code

Next lets change our kernel code so it can handle multiple Inputs. 

The first change we are going to make is to have separate offsets for the z values array and the bias array. 
Since the z values depend on the input nr while the biases don’t. 
Therefore the z values have a different data structure.
Previously we could use the same offset for both arrays but now they are different.


```diff
+__global__ void multiple_inputs(float *weight_matrix, float *biases, float *z_values, float *activation_values, int* shape, int shape_length)
 {
        int id = threadIdx.x;

        // Define offset for the current layer
-       int layer_offset_z_b = 0;
+       int layer_offset_z = 0;
+       int layer_offset_b = 0;
```

Also for convenience we will add a offset for the activations, which will be fed into the current layer. And another offset for the activations the current layer will produce.

```diff
-       int layer_offset_activations = 0;
+       int layer_offset_activations_input_layer = 0;
+       int layer_offset_activations_current_layer = shape[0] * blockDim.y;
```

This variables only take care of indexing into the correct layer but they don’t take care of individual inputs or neurons.
One detail about layer_offset_activations_current_layer we initialize it to `shape[0] * blockDim.y`

Basically we make sure that the offset points to the correct memory see picture below.
[Insert Picture here of memory layout].
![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/31b4aec6-c708-4618-8ee0-0d69fda45fe1)
Shape[0] is the number of input neurons , blockDim.y  refers to the number of threads in the y-Axis. 
Which in turn refers to the number of inputs.

The next change is we save the layer_size of the current layer in a variable so we can reuse it in the next few lines. 

```diff
        for (int shape_index = 0; shape_index < shape_length; shape_index++)
        {
                // Other threads don't execute anything to avoid out of bounds access
                if (id < shape[shape_index + 1])
                {
                        int nr_inputs_to_this_layer = shape[shape_index];
+                       int layer_size = shape[shape_index + 1];
```


So for the computation of the weighted sum. Weights * activations. We must change the indexing for the z_values and activations. The indices for the weights stay the same since they are input independent. 
For the z_values we add threadIdx.y * layer_size. Where threadIdx.y to the input the current thread has to take care off. And we multiply it with the layer_size. 

```diff
        for (int shape_index = 0; shape_index < shape_length; shape_index++)
        {
                // Other threads don't execute anything to avoid out of bounds access
                if (id < shape[shape_index + 1])
                {
                        int nr_inputs_to_this_layer = shape[shape_index];
+                       int layer_size = shape[shape_index + 1];
+
                        // w*x
                        for (int neuron_nr = 0; neuron_nr < nr_inputs_to_this_layer; neuron_nr++)
                        {
-                               z_values[layer_offset_z_b + id] += weight_matrix[layer_offset_weights + (nr_inputs_to_this_layer)* id + neuron_nr] *
-                                       activation_values[layer_offset_activations + neuron_nr];
+                               z_values[layer_offset_z + threadIdx.y * layer_size + id] += weight_matrix[layer_offset_weights + (nr_inputs_to_this_layer)* id + neuron_nr] *
+                                       activation_values[layer_offset_activations_input_layer + threadIdx.y * nr_inputs_to_this_layer + neuron_nr];
```

Here are a few images to illustrate the indexing logic:  

![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/56cf7b3e-6607-4a88-b894-e219622c36bf)
All the threads with threadyId.y = 0 will take care of the first input because 0 * 6 = 0 which is the beginning of the array.

![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/2acdba78-9cce-4f1a-8191-d553c39fa27e)
The threads where the threadIdx.y variable = 1 will take care of the second input.
![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/d51f6eea-9bc7-4c45-9aa1-6b353317ca10)
Now when we move on to the next layer in  the neural network. We will have layer_offset_z point to the z values of the next layer. Apart from the offset pointing moving us into the correct layer the indexing logic stays the same. The threadIdx.y variable will help to point us towards the correct input. With the same formula threadIdx * layer size, obviously the layer size is now different since we have 4 neurons instead of 6,  like in the first layer.

![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/48b1fda7-02a2-43f9-ae74-65bff12691b7)


For the activation_values array which contains the inputs to the current layer we add `threadIdx.y * nr_inputs_to_this_layer` to the index. 
```diff
-                               z_values[layer_offset_z_b + id] += weight_matrix[layer_offset_weights + (nr_inputs_to_this_layer)* id + neuron_nr] *
-                                       activation_values[layer_offset_activations + neuron_nr];
+                               z_values[layer_offset_z + threadIdx.y * layer_size + id] += weight_matrix[layer_offset_weights + (nr_inputs_to_this_layer)* id + neuron_nr] *
+                                       activation_values[layer_offset_activations_input_layer + threadIdx.y * nr_inputs_to_this_layer + neuron_nr];
```

To finish the z_value computation we also need to add the bias. The indexing for the z_value follows the same logic as in the line above. The indexing into the bias array is the same as in the previous video.  

```diff
                        // w*x + b
-                       z_values[layer_offset_z_b + id] += biases[layer_offset_z_b + id];
+                       z_values[layer_offset_z + threadIdx.y * layer_size + id] += biases[layer_offset_b + id];
```

Alright so the last step is to compute the sigmoid function.

```diff
-                       
-                       activation_values[layer_offset_activations + shape[shape_index] + id] = 1.0 / (1.0 + exp(-z_values[layer_offset_z_b + id]));
+                       activation_values[layer_offset_activations_current_layer + layer_size * threadIdx.y + id] = 1.0 / (1.0 + exp(-z_values[layer_offset_z + layer_size * threadIdx.y + id]));
                }
```
So the indexing logic for the z_values stays the same as in the code above. And for the activations it’s very similar too. 
The difference is just that we use the activations_current_layer_offset variable instead of the layer_offset_z variable.
That’s it for the computation. 


At the end we update all our layer_offset variables.
![layer_offset_updates](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/6fa3be9e-0cb7-4562-a1db-e9997572c3a6)

Basically we just change the all layer_offsets so they point one layer further. 
We do this for the weights, biases, z_values , and activations offsets.
You can see this illustrated in this image.
That’s it for the CUDA Kernel. 


## 4.1 Verifiying results

Now let’s run and test our code.  We have the python code for verification. Which does the neural network forward pass with numpy. We use the exact same weights and biases as in the cuda code. And also we use the same inputs. If the CUDA code produces the same results as our Python verification program. We can be sure that our CUDA code is correct. 

### 4.1.1 Python Numpy Reference Code

```python
import numpy as np

nr_inputs = 3
shape = np.array([8, 6, 4, 1])
weights = np.array([1.62, -0.61, -0.53, -1.07, 0.87, -2.30, 1.74, -0.76, 0.32, -0.25, 1.46, -2.06, -0.32, -0.38, 1.13,
                       -1.10, -0.17, -0.88, 0.04, 0.58, -1.10, 1.14, 0.90, 0.50, 0.90, -0.68, -0.12, -0.94, -0.27, 0.53,
                       -0.69, -0.40, -0.69, -0.85, -0.67, -0.01, -1.12, 0.23, 1.66, 0.74, -0.19, -0.89, -0.75, 1.69, 0.05,
                       -0.64, 0.19, 2.10, 0.12, 0.62, 0.30, -0.35, -1.14, -0.35, -0.21, 0.59, 0.84, 0.93, 0.29, 0.89, -0.75,
                       1.25, 0.51, -0.30, 0.49, -0.08, 1.13, 1.52, 2.19, -1.40, -1.44, -0.50, 0.16, 0.88, 0.32, -2.02])

biases = np.array([-0.31, 0.83, 0.23, 0.76, -0.22, -0.20, 0.19, 0.41, 0.20, 0.12, -0.67])

act = np.array([0.38, 0.12, 1.13, 1.20, 0.19, -0.38, -0.64, 0.42, 0.76, -0.36, -0.23, -0.89, -0.01, -0.08, -0.26, -0.13, -0.55, -0.42, -0.39, -0.83, 0.87, 0.44, -0.45, -0.52])
activations = [act.reshape(shape[0], nr_inputs, order="F"),
               np.zeros((shape[1], nr_inputs)),
               np.zeros((shape[2], nr_inputs)),
               np.zeros((shape[3], nr_inputs))
               ]

weights = [weights[:48].reshape(6, 8), weights[48:72].reshape(4, 6), weights[72:].reshape(1, 4)]
biases = [biases[:6].reshape(-1, 1), biases[6:10].reshape(-1, 1), biases[10:].reshape(-1, 1)]


def sig(z):
    return 1.0/(1.0+np.exp(-z))


z_values = []

for layer_index in range(3):
  z = np.dot(weights[layer_index], activations[layer_index]) + biases[layer_index]
  z_values.append(z)
  activations[layer_index + 1] += sig(z)

for layer_index in range(3):
    print(f"Z Values {layer_index + 1}. hidden layer")
    print(z_values[layer_index])

for layer_index in range(1, 4):
    print(f"Activation Values {layer_index}. hidden layer")
    print(activations[layer_index])

```

### 4.1.2 Result comparision

![image](https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/assets/16619270/47f4f7db-473c-4d36-8abd-65cd2b79e367)

Yes, the values are the same. So this means our CUDA implementation is correct. 

## 5. FULL DIFF

```diff
--- a/Part 3 - Multiple Layers/kernel.cu
+++ b/Part 4 - Multiple Inputs/kernel.cu
@@ -1,162 +1,180 @@
 #include <device_functions.h>
 #include "cuda_runtime.h"
-#include "cuda_runtime_api.h"
 #include "device_launch_parameters.h"
 #include <math.h>
 #include <stdio.h>
 #include <string>
 #include <iostream>
 #include <algorithm>


 #pragma once
 #ifdef __INTELLISENSE__
 void __syncthreads();
 #endif


-__global__ void linear_layer_and_activation(float *weight_matrix, float *biases,
-                                                                                       float *z_values, float *activation_values,
-                                                                                       int* shape, int shape_length)
+__global__ void multiple_inputs(float *weight_matrix, float *biases, float *z_values, float *activation_values, int* shape, int shape_length)
 {
        int id = threadIdx.x;

        // Define offset for the current layer
-       int layer_offset_z_b = 0;
+       int layer_offset_z = 0;
+       int layer_offset_b = 0;
        int layer_offset_weights = 0;
-       int layer_offset_activations = 0;
+       int layer_offset_activations_input_layer = 0;
+       int layer_offset_activations_current_layer = shape[0] * blockDim.y;

        for (int shape_index = 0; shape_index < shape_length; shape_index++)
        {
                // Other threads don't execute anything to avoid out of bounds access
                if (id < shape[shape_index + 1])
                {
                        int nr_inputs_to_this_layer = shape[shape_index];
+                       int layer_size = shape[shape_index + 1];
+
                        // w*x
                        for (int neuron_nr = 0; neuron_nr < nr_inputs_to_this_layer; neuron_nr++)
                        {
-                               z_values[layer_offset_z_b + id] += weight_matrix[layer_offset_weights + (nr_inputs_to_this_layer)* id + neuron_nr] *
-                                       activation_values[layer_offset_activations + neuron_nr];
+                               z_values[layer_offset_z + threadIdx.y * layer_size + id] += weight_matrix[layer_offset_weights + (nr_inputs_to_this_layer)* id + neuron_nr] *
+                                       activation_values[layer_offset_activations_input_layer + threadIdx.y * nr_inputs_to_this_layer + neuron_nr];
                        }

                        // w*x + b
-                       z_values[layer_offset_z_b + id] += biases[layer_offset_z_b + id];
+                       z_values[layer_offset_z + threadIdx.y * layer_size + id] += biases[layer_offset_b + id];

                        // sig(w*x + b)
-                       //                                      + shape[shape_index] => write activation values for next layer,
-                       //                                                      instead of overwriting the input values
-                       activation_values[layer_offset_activations + shape[shape_index] + id] = 1.0 / (1.0 + exp(-z_values[layer_offset_z_b + id]));
+                       // + shape[shape_index] => write activation values for next layer,instead of overwriting the input values
+                       activation_values[layer_offset_activations_current_layer + layer_size * threadIdx.y + id] = 1.0 / (1.0 + exp(-z_values[layer_offset_z + layer_size * threadIdx.y + id]));
                }

                // Important to do this outside the Memory Guard
                layer_offset_weights += shape[shape_index] * shape[shape_index + 1];
-               layer_offset_z_b += shape[shape_index + 1];
-               layer_offset_activations += shape[shape_index];
+               layer_offset_b += shape[shape_index + 1];
+               layer_offset_z += shape[shape_index + 1] * blockDim.y;
+               layer_offset_activations_input_layer = layer_offset_activations_current_layer;
+               layer_offset_activations_current_layer += shape[shape_index + 1] * blockDim.y;

-               // Call syncthreads so we know that all threads have finished working on the current layerbefore we take care of the next layer
+               // Call syncthreads so we know that all threads have finished working on the current layer before we take care of the next layer
                // Try removing this and guess what will happen.
                __syncthreads();
        }
 }

 int main()
 {
        const int shape_length = 4;
        int shape[shape_length] = { 8, 6, 4, 1 };
+       const int NR_INPUTS = 3;

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

-       // The first 8 values are our inputs rest of the array can be initialized with 0.0
-       float *host_activations = new float [nr_neurons] {0.38f, 0.12f, 1.13f, 1.20f, 0.19f, -0.38f, -0.64f, 0.42f};
+       // The first 8 values are our inputs rest of the array can be initialized with 0.0
+       int nr_activations = nr_neurons * NR_INPUTS;
+       float *host_activations = new float [nr_activations] {0.38f, 0.12f, 1.13f, 1.20f, 0.19f, -0.38f, -0.64f, 0.42f, 0.76f, -0.36f, -0.23f, -0.89f, -0.01f, -0.08f, -0.26f, -0.13f, -0.55f, -0.42f, -0.39f, -0.83f, 0.87f, 0.44f, -0.45f, -0.52f};

        // Initialize z Matrix
-       float *host_z = new float [nr_biases] {0.0f};
+       int nr_z = nr_biases * NR_INPUTS;
+       float *host_z = new float [nr_z] {0.0f};


        // Calculate the amount of memory needed so we can provide this information to cuda malloc
        const size_t bytes_biases = nr_biases * sizeof(float);
-       const size_t bytes_z = nr_biases * sizeof(float);
+       const size_t bytes_z = nr_z * sizeof(float);
        const size_t bytes_weights = nr_weights * sizeof(float);
-       const size_t bytes_activations = nr_neurons * sizeof(float);
+       const size_t bytes_activations = nr_activations * sizeof(float);
        const size_t bytes_shape = sizeof(int) * shape_length;


        // Allocate GPU device memory
        float *d_biases, *d_weights, *d_activations, *d_z;
        int *d_shape;
        cudaMalloc(&d_biases, bytes_biases);
        cudaMalloc(&d_weights, bytes_weights);
        cudaMalloc(&d_activations, bytes_activations);
        cudaMalloc(&d_z, bytes_z);
        cudaMalloc(&d_shape, bytes_shape);

        // Copy data from CPU Memory to GPU Memory
        cudaMemcpy(d_biases, host_biases, bytes_biases, cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, host_weights, bytes_weights, cudaMemcpyHostToDevice);
        cudaMemcpy(d_activations, host_activations, bytes_activations, cudaMemcpyHostToDevice);
        cudaMemcpy(d_z, host_z, bytes_z, cudaMemcpyHostToDevice);
        cudaMemcpy(d_shape, shape, bytes_shape, cudaMemcpyHostToDevice);

        // Call cuda kernel
-       int nr_threads = *std::max_element(shape, shape + shape_length);
-       linear_layer_and_activation << <1, nr_threads >> > (d_weights, d_biases, d_z, d_activations, d_shape, shape_length);
+       int nr_threads_x_dimension = *std::max_element(shape + 1, shape + shape_length);
+       dim3 thread_block_dimensions(nr_threads_x_dimension, NR_INPUTS);
+       multiple_inputs << <1, thread_block_dimensions >> > (d_weights, d_biases, d_z, d_activations, d_shape, shape_length);

        // After we caclulated the activations and z values we need to copy the data from GPU Memory back to the CPU Memory
        cudaMemcpy(host_activations, d_activations, bytes_activations, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_z, d_z, bytes_z, cudaMemcpyDeviceToHost);

        // Free our memory
        cudaFree(d_biases);
        cudaFree(d_weights);
        cudaFree(d_activations);
        cudaFree(d_z);
        cudaFree(d_shape);

        int z_offset = 0;
        for (int shape_index = 1; shape_index < shape_length; shape_index++)
        {
                std::cout << "Z Values " << shape_index << ". hidden layer" << std::endl;
+
                for (int neuron_nr = 0; neuron_nr < shape[shape_index]; neuron_nr++)
                {
-                       std::cout << host_z[neuron_nr + z_offset] << std::endl;
+                       std::cout << "[";
+                       for (int input = 0; input < NR_INPUTS; input++)
+                       {
+                               std::cout << host_z[z_offset + input * shape[shape_index] + neuron_nr] << ", ";
+                       }
+                       std::cout << "]" << std::endl;
+
                }
-               z_offset += shape[shape_index];
+               z_offset += shape[shape_index] * NR_INPUTS;
        }

-       int activations_offset = shape[0]; // Skip input values
+       int activations_offset = shape[0] * NR_INPUTS; // Skip input values
        for (int shape_index = 1; shape_index < shape_length; shape_index++)
        {
                std::cout << "Activations " << shape_index << ". hidden layer" << std::endl;

                for (int neuron_nr = 0; neuron_nr < shape[shape_index]; neuron_nr++)
                {
-                       std::cout << host_activations[neuron_nr + activations_offset] << std::endl;
+                       std::cout << "[";
+                       for (int input = 0; input < NR_INPUTS; input++)
+                       {
+                               std::cout << host_activations[activations_offset + input * shape[shape_index] + neuron_nr] << ", ";
+                       }
+                       std::cout << "]" << std::endl;
                }
-               activations_offset += shape[shape_index];
+               activations_offset += shape[shape_index] * NR_INPUTS;
        }

        getchar();


        return;
}
```
