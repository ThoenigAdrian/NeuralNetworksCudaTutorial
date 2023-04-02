## Multiple Layers


### Cuda kernel Code




```c
__global__ void linear_layer_and_activation(float *weight_matrix, float *biases, float *x_inputs, 
	                                    float *z_values, float *activation_values, 
					    int nr_output_neurons, int nr_input_neurons)
{

	
}
```


## Defining the shape


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


## 2. Request GPU Memory for the shape 

First we calculate the numbers of bytes needed because we are going to need this for the cudaMalloc call.


## 4. Launching the Kernel - Changing the paraemters

Alright that’s all the code we need for the kernel. We now have a kernel which can compute all the z values and activations. The only thing left to do now is to call the kernel. This can be done via the triple chevron launch syntax. 

```c
// Call the kernel which calculates the activations.
// Call cuda kernel
linear_layer_and_activation << <1 , nr_threads >> > (d_weights, d_biases, d_inputs, d_z, d_activations, d_shape, shape_length);
```


## 6. Print the results
```c

```

### [Full code](kernel.cu)
