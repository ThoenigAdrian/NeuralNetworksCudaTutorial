#include <device_functions.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
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


__global__ void linear_layer_and_activation(float *weight_matrix, float *biases, 
											float *z_values, float *activation_values,
											int* shape, int shape_length)
{
	int id = threadIdx.x;

	// Define offset for the current layer
	int layer_offset_z_b = 0;
	int layer_offset_weights = 0;
	int layer_offset_activations = 0;
	
	for (int shape_index = 0; shape_index < shape_length; shape_index++)
	{
		// Other threads don't execute anything to avoid out of bounds access
		if (id < shape[shape_index + 1])
		{
			int nr_inputs_to_this_layer = shape[shape_index];
			// w*x
			for (int neuron_nr = 0; neuron_nr < nr_inputs_to_this_layer; neuron_nr++)
			{
				z_values[layer_offset_z_b + id] += weight_matrix[layer_offset_weights + (nr_inputs_to_this_layer)* id + neuron_nr] *
					activation_values[layer_offset_activations + neuron_nr];
			}

			// w*x + b
			z_values[layer_offset_z_b + id] += biases[layer_offset_z_b + id];

			// sig(w*x + b)	
			// 		                        + shape[shape_index] => write activation values for next layer,
			//                                                      instead of overwriting the input values
			activation_values[layer_offset_activations + shape[shape_index] + id] = 1.0 / (1.0 + exp(-z_values[layer_offset_z_b + id]));
		}

		// Important to do this outside the Memory Guard 
		layer_offset_weights += shape[shape_index] * shape[shape_index + 1];
		layer_offset_z_b += shape[shape_index + 1];
		layer_offset_activations += shape[shape_index];

		// Call syncthreads so we know that all threads have finished working on the current layerbefore we take care of the next layer
		// Try removing this and guess what will happen.
		__syncthreads(); 
	}
}

int main()
{
	const int shape_length = 4;
	int shape[shape_length] = { 8, 6, 4, 1 };

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
	
	// The first 8 values are our inputs rest of the array can be initialized with 0.0 
	float *host_activations = new float [nr_neurons] {0.38f, 0.12f, 1.13f, 1.20f, 0.19f, -0.38f, -0.64f, 0.42f};
	
	// Initialize z Matrix
	float *host_z = new float [nr_biases] {0.0f};


	// Calculate the amount of memory needed so we can provide this information to cuda malloc
	const size_t bytes_biases = nr_biases * sizeof(float);
	const size_t bytes_z = nr_biases * sizeof(float);
	const size_t bytes_weights = nr_weights * sizeof(float);
	const size_t bytes_activations = nr_neurons * sizeof(float);
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
	int nr_threads = *std::max_element(shape, shape + shape_length);
	linear_layer_and_activation << <1, nr_threads >> > (d_weights, d_biases, d_z, d_activations, d_shape, shape_length);

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
		for (int neuron_nr = 0; neuron_nr < shape[shape_index]; neuron_nr++)
		{
			std::cout << host_z[neuron_nr + z_offset] << std::endl;
		}
		z_offset += shape[shape_index];
	}

	int activations_offset = shape[0]; // Skip input values	
	for (int shape_index = 1; shape_index < shape_length; shape_index++)
	{
		std::cout << "Activations " << shape_index << ". hidden layer" << std::endl;

		for (int neuron_nr = 0; neuron_nr < shape[shape_index]; neuron_nr++)
		{
			std::cout << host_activations[neuron_nr + activations_offset] << std::endl;
		}
		activations_offset += shape[shape_index];
	}

	getchar();


	return 0;
}