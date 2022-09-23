#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <string>
#include <iostream>


__global__ void linear_layer_and_activation(float *weight_matrix, float *biases, float *x_inputs, 
	                                        float *z_values, float *activation_values, 
											int nr_output_neurons, int nr_input_neurons)
{
	int id = threadIdx.x;

	// w*x
	for (int neuron_nr = 0; neuron_nr < nr_input_neurons; neuron_nr++)
	{
		z_values[id] += weight_matrix[(nr_input_neurons)* id + neuron_nr] * x_inputs[neuron_nr];
	}

	// w*x + b
	z_values[id] += biases[id];

	// sig(w*x + b)
	activation_values[id] = 1.0 / (1.0 + exp(-z_values[id]));
	
}

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


	// Calculate the amount of memory needed so we can provide this information to cuda malloc
	const size_t bytes_biases = size_b * sizeof(float);
	const size_t bytes_z = size_b * sizeof(float);
	const size_t bytes_weights = size_w * sizeof(float);
	const size_t bytes_activations = size_b * sizeof(float);
	const size_t bytes_inputs = INPUT_NEURONS * sizeof(float);


	// Allocate GPU device memory
	float *d_biases, *d_weights, *d_activations, *d_z, *d_inputs;
	cudaMalloc(&d_biases, bytes_biases);
	cudaMalloc(&d_weights, bytes_weights);
	cudaMalloc(&d_activations, bytes_activations);
	cudaMalloc(&d_z, bytes_z);
	cudaMalloc(&d_inputs, bytes_inputs);


	// Copy data from CPU Memory to GPU Memory
	cudaMemcpy(d_biases, host_biases, bytes_biases, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, host_weights, bytes_weights, cudaMemcpyHostToDevice);
	cudaMemcpy(d_activations, host_activations, bytes_activations, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, host_z, bytes_z, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inputs, host_input, bytes_inputs, cudaMemcpyHostToDevice);

	// Call cuda kernel
	linear_layer_and_activation << <OUTPUT_NEURONS / 256 + 1, OUTPUT_NEURONS >> > (d_weights, d_biases, d_inputs, d_z, d_activations, OUTPUT_NEURONS, INPUT_NEURONS);

	// After we caclulated the activations and z values we need to copy the data from GPU Memory back to the CPU Memory
	cudaMemcpy(host_activations, d_activations, bytes_activations, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_z, d_z, bytes_z, cudaMemcpyDeviceToHost);

	// Free our memory
	cudaFree(d_biases);
	cudaFree(d_weights);
	cudaFree(d_activations);
	cudaFree(d_z);


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

	getchar();


	return 0;
}
