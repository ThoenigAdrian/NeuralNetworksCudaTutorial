## Linear Layer

Last time we assumed a given **z value** in order to be able to compute the activation function.
This time we are actually going to compute the **z value** , also known as weighted sum.
In neural networks the thing which computes the weighted sum is the **Linear Layer**.

The first thing we need before we start is a plan on how we should exploit the many threads we can use.
So we need to come up with a scheme on how to distribute the work among the threads.

We are going to use the follwoing idea:

One thread takes care of one neuron. So if we have 3 output neurons we are going to use 3 threads.
Each of this thread uses the weights which belongs to it's assigned output neuron and calculates the weighted sum and then calculates the activation value.

![](threadtimeline.png)



### Cuda kernel Code


For the cuda kernel we need the following **input information**
1. The values of the weights
2. The values of the biases
3. The values of the inputs
4. The number of input neurons
5. The number of output neurons

And we need to calculate the **follwoing quantites**
1. z_values
2. activation_values


```c
__global__ void linear_layer_and_activation(float *weight_matrix, float *biases, float *x_inputs, 
	                                    float *z_values, float *activation_values, 
					    int nr_output_neurons, int nr_input_neurons)
{
        // We use the thread id so we can index different weights/neurons in each thread
	int id = threadIdx.x;

	// w*x
	// We loop over every incoming neuron and multiply it with it's weight and sum it to the current z_value
	for (int neuron_nr = 0; neuron_nr < nr_input_neurons; neuron_nr++)
	{
		z_values[id] += weight_matrix[(nr_input_neurons)* id + neuron_nr] * x_inputs[neuron_nr];
	}

	// w*x + b
	// Don't forget to add the bias
	z_values[id] += biases[id];

	// The we compute the activation value just like in part1 of the tutorial
	// sig(w*x + b)
	activation_values[id] = 1.0 / (1.0 + exp(-z_values[id]));
	
}
```

## 1. CPU Buffers

The first thing we are going to do is to create all the needed buffers. 

1. We need one buffer on the CPU for the z values 
2. And one for the activation values. 

We will call this host_z_values and host_activations. In the cuda world it’s convention to designate data which is stored on **cpu** accessible memory with **host**. And data which is stored on **GPU** accessible memory with **device**. 

```c
int main()
{
  const int INPUT_NEURONS = 4;
	const int OUTPUT_NEURONS = 3;

	// Initialize weights on CPU/RAM
	const int size_w = INPUT_NEURONS * OUTPUT_NEURONS;
	float *host_weights = new float [size_w] {0.80f, 0.87f, 0.16f, 0.96f, 0.89f, 0.87f, 0.31f, 0.08f, 0.09f, 0.69f, 0.03f, 0.42f};

	// Initialize biases on CPU/RAM
	const int size_b = OUTPUT_NEURONS;
	float *host_biases = new float [size_b] {0.68f, 0.83f, 0.01f};
	float host_input[INPUT_NEURONS] = { 0.75f,0.98f,0.74f,0.28f };

	// Initialize activations on CPU/RAM
	float *host_activations = new float [size_b] {0.0, 0.0, 0.0};

	// Initialize z Matrix
	float *host_z = new float [size_b] {0.0, 0.0, 0.0};
}
```


## 2. Request GPU Memory

First we calculate the numbers of bytes needed because we are going to need this for the cudaMalloc call.

```c
// Calculate the amount of memory needed so we can provide this information to cuda malloc
const size_t bytes_biases = size_b * sizeof(float);
const size_t bytes_z = size_b * sizeof(float);
const size_t bytes_weights = size_w * sizeof(float);
const size_t bytes_activations = size_b * sizeof(float);
const size_t bytes_inputs = INPUT_NEURONS * sizeof(float);
```
Now we can call cudaMalloc, cudaMalloc expects two arguments, the first is the address of a pointer, the second is the number of bytes to allocate. cudaMalloc will change the value of the pointer so it points to free GPU memory with as much memory as we requested.
```c
// Allocate GPU device memory
float *d_biases, *d_weights, *d_activations, *d_z, *d_inputs;
cudaMalloc(&d_biases, bytes_biases);
cudaMalloc(&d_weights, bytes_weights);
cudaMalloc(&d_activations, bytes_activations);
cudaMalloc(&d_z, bytes_z);
cudaMalloc(&d_inputs, bytes_inputs);
```
## 3. Transfer z Values from CPU to GPU

This can be done with cudaMemcpy. 
The function signature is as follows:
1. destination buffer
2. source buffer
3. number of bytes
4. direction. 

For the direction there are 2 options: ```cudaMemcpyHostToDevice``` and ```cudaMemcpyDeviceToHost```

```c
// Copy data from CPU Memory to GPU Memory
cudaMemcpy(d_biases, host_biases, bytes_biases, cudaMemcpyHostToDevice);
cudaMemcpy(d_weights, host_weights, bytes_weights, cudaMemcpyHostToDevice);
cudaMemcpy(d_activations, host_activations, bytes_activations, cudaMemcpyHostToDevice);
cudaMemcpy(d_z, host_z, bytes_z, cudaMemcpyHostToDevice);
cudaMemcpy(d_inputs, host_input, bytes_inputs, cudaMemcpyHostToDevice);
```


## 5. Launching the Kernel

Alright that’s all the code we need for the kernel. We now have a kernel which can compute all the z values and activations. The only thing left to do now is to call the kernel. This can be done via the triple chevron launch syntax. 

```c
// Call the kernel which calculates the activations.
// 1 = Number of Blocks
// OUTPUT_NEURONS is the number of threads. Since for each neuron we need 1 thread
// Call cuda kernel
linear_layer_and_activation << <OUTPUT_NEURONS / 256 + 1, OUTPUT_NEURONS >> > (d_weights, d_biases, d_inputs, d_z, d_activations, OUTPUT_NEURONS, INPUT_NEURONS);
```

## 6. Transfering computed values back to CPU

While we now computed the activation values we still can't access them because they are still only available on GPU accessible memory.
In order to be able to print them we need to transfer them to the CPU. 
```c
// After we caclulated the activations and z values we need to copy the data from GPU Memory back to the CPU Memory
cudaMemcpy(host_activations, d_activations, bytes_activations, cudaMemcpyDeviceToHost);
cudaMemcpy(host_z, d_z, bytes_z, cudaMemcpyDeviceToHost);
```

## 7. Print the results
```c
std::cout << "Z Values: " << std::endl;
for (int neuron_nr = 0; neuron_nr < OUTPUT_NEURONS; neuron_nr++)
{
	std::cout << host_z[neuron_nr] << std::endl;
}


std::cout << std::endl << "Activations: " << std::endl;
for (int neuron_nr = 0; neuron_nr < OUTPUT_NEURONS; neuron_nr++)
{
	std::cout << host_activations[neuron_nr] << std::endl;
}
```

```
Z Values: 
[[2.5198]
 [2.6019]
 [0.8935]]

Activations: 
[[0.92551827]
 [0.93098376]
 [0.70961192]]
```

[Full code](kernel.cu)
