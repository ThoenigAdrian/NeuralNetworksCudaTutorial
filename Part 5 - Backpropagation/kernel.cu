#include <device_functions.h>
#include "cuda_runtime.h"
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

 
__global__ void feed_forward_and_backprop(float *weight_matrix, float *biases, float *z_values, float *activation_values, int* shape, int shape_length, int* gradients_offset_table, float* gradients, float* gradients_weights, float* target,
	int* activation_offset_table, int* wm_offset_table)
{
	int id = threadIdx.x;

	// Define offset for the current layer
	int output_neurons = shape[shape_length - 1];
	int layer_offset_z = 0;
	int layer_offset_b = 0;
	int layer_offset_weights = 0;
	int layer_offset_activations_input_layer = 0;
	int layer_offset_activations_current_layer = shape[0] * gridDim.x;
	
	for (int shape_index = 0; shape_index < shape_length - 1; shape_index++)
	{
		// Other threads don't execute anything to avoid out of bounds access
		if (id < shape[shape_index + 1])
		{
			int nr_inputs_to_this_layer = shape[shape_index];
			int layer_size = shape[shape_index + 1];
			
			// w*x
			for (int neuron_nr = 0; neuron_nr < nr_inputs_to_this_layer; neuron_nr++)
			{
				z_values[layer_offset_z + blockIdx.x * layer_size + id] += weight_matrix[layer_offset_weights + (nr_inputs_to_this_layer)* id + neuron_nr] *
					activation_values[layer_offset_activations_input_layer + blockIdx.x * nr_inputs_to_this_layer + neuron_nr];
			}

			// w*x + b
			z_values[layer_offset_z + blockIdx.x * layer_size + id] += biases[layer_offset_b + id];

			// sig(w*x + b)	
			// + shape[shape_index] => write activation values for next layer,instead of overwriting the input values                                                
			activation_values[layer_offset_activations_current_layer + layer_size * blockIdx.x + id] = 1.0 / (1.0 + exp(-z_values[layer_offset_z + layer_size * blockIdx.x + id]));
		}

		// Important to do this outside the Memory Guard 
		layer_offset_weights += shape[shape_index] * shape[shape_index + 1];
		layer_offset_b += shape[shape_index + 1];
		layer_offset_z += shape[shape_index + 1] * gridDim.x;
		layer_offset_activations_input_layer = layer_offset_activations_current_layer;
		layer_offset_activations_current_layer += shape[shape_index + 1] * gridDim.x;

		// Call syncthreads so we know that all threads have finished working on the current layer before we take care of the next layer
		// Try removing this and guess what will happen.
		__syncthreads(); 
	}

    if(threadIdx.x < shape[shape_length -1])
    {
		gradients[gradients_offset_table[shape_length-2] + blockIdx.x * output_neurons + threadIdx.x] = activation_values[activation_offset_table[shape_length-1] + blockIdx.x * output_neurons + threadIdx.x] - 
			target[blockIdx.x * output_neurons + threadIdx.x];
    }
    __syncthreads();
	
	
	for (int layer = shape_length - 2; layer > 0; layer--)
	{
		// BP 2 + 3
		if (threadIdx.x < shape[layer])
		{
            float grad = 0.0f;
			int c = gradients_offset_table[layer - 1] + blockIdx.x * shape[layer] + threadIdx.x;;
			for (int higher_layer_neuron_nr = 0; higher_layer_neuron_nr < shape[layer+1]; higher_layer_neuron_nr++)
			{
				int a = gradients_offset_table[layer] + higher_layer_neuron_nr + blockIdx.x * shape[layer + 1];
				int b = wm_offset_table[layer] + higher_layer_neuron_nr * shape[layer] + threadIdx.x;
				grad += gradients[a] * weight_matrix[b];
			}
            float activation = activation_values[activation_offset_table[layer] + threadIdx.x + blockIdx.x * shape[layer]];
			float sig_strich = activation * (1 - activation);
			gradients[c] = grad * sig_strich;
		}
		__syncthreads();
	}	
	
	// BP4 
	for (int layer = 0; layer < shape_length - 1; layer++)
	{
		int nr_weights = shape[layer] * shape[layer + 1];
		// for each weight one thread
		if (threadIdx.x < nr_weights)
		{			
			gradients_weights[wm_offset_table[layer] * gridDim.x + blockIdx.x * nr_weights + threadIdx.x] = 
				activation_values[activation_offset_table[layer] + blockIdx.x * shape[layer] + threadIdx.x % shape[layer]] * 
				gradients[gradients_offset_table[layer] + blockIdx.x * shape[layer + 1] + threadIdx.x / shape[layer]];
		}
		
	}
}

void calculate_offsets(int* activation_offset_table, int* gradient_offset_table, int* weight_matrix_offset_table, int* shape, int shape_length, int nr_of_inputs)
{
	// Goal of this to get the correct offset for a certain layer by offset_table[layer]
	activation_offset_table[0] = 0;
	gradient_offset_table[0] = 0;
	weight_matrix_offset_table[0] = 0;
	for (int i = 1; i < shape_length; i++)
	{
		activation_offset_table[i] = activation_offset_table[i - 1] + shape[i - 1] * nr_of_inputs;
	}
	for (int i = 1; i < shape_length - 1; i++)
	{
		gradient_offset_table[i] = gradient_offset_table[i - 1] + shape[i] * nr_of_inputs;
		weight_matrix_offset_table[i] = weight_matrix_offset_table[i - 1] + shape[i] * shape[i - 1];
	}
}

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
	float *host_target = new float [NR_INPUTS * shape[shape_length - 1]] {0.78, 0.64, 0.55}; // This are the values we want to the network to produce after training , basically our "labels" we need them so we can compute any gradients at all just set them 0.3 and 0.6 no special meaning

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
	int nr_activations = nr_neurons * NR_INPUTS;
	float *host_activations = new float [nr_activations] {0.38f, 0.12f, 1.13f, 1.20f, 0.19f, -0.38f, -0.64f, 0.42f, 0.76f, -0.36f, -0.23f, -0.89f, -0.01f, -0.08f, -0.26f, -0.13f, -0.55f, -0.42f, -0.39f, -0.83f, 0.87f, 0.44f, -0.45f, -0.52f};
	
	// Initialize z Matrix
	int nr_z = nr_biases * NR_INPUTS;
	int* host_activation_offset_table = new int[shape_length];
	int* host_wm_offset_table = new int[shape_length-1];
	int* host_gradient_offset_table = new int[shape_length - 1];

	calculate_offsets(host_activation_offset_table, host_gradient_offset_table, host_wm_offset_table, shape, shape_length, NR_INPUTS);

	float *host_z = new float [nr_z] {0.0f};
	float *host_gradients = new float[nr_biases * NR_INPUTS] {0.0f};
	float *host_gradients_weights = new float[nr_weights * NR_INPUTS] {0.0f};


	// Calculate the amount of memory needed so we can provide this information to cuda malloc
	const size_t bytes_biases = nr_biases * sizeof(float);
	const size_t bytes_z = nr_z * sizeof(float);
	const size_t bytes_weights = nr_weights * sizeof(float);
	const size_t bytes_activations = nr_activations * sizeof(float);
	const size_t bytes_shape = sizeof(int) * shape_length;
	const size_t bytes_gradients = sizeof(float) * nr_biases * NR_INPUTS;
	const size_t bytes_gradients_weights = sizeof(float) * nr_weights * NR_INPUTS;
	const size_t bytes_activation_offset_table = sizeof(float) * shape_length;
	const size_t bytes_wm_offset_table = sizeof(float) * (shape_length - 1);
	const size_t bytes_gradient_offset_table = sizeof(float) * (shape_length - 1);
	const size_t bytes_target = sizeof(float) * shape[shape_length - 1] * NR_INPUTS;


	// Allocate GPU device memory
	float *d_biases, *d_weights, *d_activations, *d_z, *d_gradients, *d_gradients_weights, *d_target;
	int *d_shape, * d_activation_offset_table, * d_wm_offset_table, * d_gradient_offset_table;
	cudaMalloc(&d_biases, bytes_biases);
	cudaMalloc(&d_weights, bytes_weights);
	cudaMalloc(&d_activations, bytes_activations);
	cudaMalloc(&d_z, bytes_z);
	cudaMalloc(&d_shape, bytes_shape);
	cudaMalloc(&d_gradients, bytes_gradients);
	cudaMalloc(&d_gradients_weights, bytes_gradients_weights);
	cudaMalloc(&d_activation_offset_table, bytes_activation_offset_table);
	cudaMalloc(&d_wm_offset_table, bytes_wm_offset_table);
	cudaMalloc(&d_gradient_offset_table, bytes_gradient_offset_table);
	cudaMalloc(&d_target, bytes_target);
	

	// Copy data from CPU Memory to GPU Memory
	cudaMemcpy(d_biases, host_biases, bytes_biases, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, host_weights, bytes_weights, cudaMemcpyHostToDevice);
	cudaMemcpy(d_activations, host_activations, bytes_activations, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, host_z, bytes_z, cudaMemcpyHostToDevice);
	cudaMemcpy(d_shape, shape, bytes_shape, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gradients, host_gradients, bytes_gradients, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradients_weights, host_gradients_weights, bytes_gradients_weights, cudaMemcpyHostToDevice);
	cudaMemcpy(d_target, host_target, bytes_target, cudaMemcpyHostToDevice);
	cudaMemcpy(d_activation_offset_table, host_activation_offset_table, bytes_activation_offset_table, cudaMemcpyHostToDevice);
	cudaMemcpy(d_wm_offset_table, host_wm_offset_table, bytes_wm_offset_table, cudaMemcpyHostToDevice);
	cudaMemcpy(d_gradient_offset_table, host_gradient_offset_table, bytes_gradient_offset_table, cudaMemcpyHostToDevice);

	// Call cuda kernel
	int nr_threads_x_dimension = *std::max_element(shape + 1, shape + shape_length);
	dim3 thread_block_dimensions(512, 1, 1);
	dim3 grid_dimensions(NR_INPUTS, 1, 1);
	feed_forward_and_backprop << <grid_dimensions, thread_block_dimensions >> > (d_weights,
		d_biases, d_z, d_activations, d_shape, shape_length,
		d_gradient_offset_table, d_gradients, d_gradients_weights, d_target,
		d_activation_offset_table, d_wm_offset_table);
	// quick debug check
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
	}

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA runtime error after launch: %s\n", cudaGetErrorString(err));
	}
	// After we caclulated the activations and z values we need to copy the data from GPU Memory back to the CPU Memory
	cudaMemcpy(host_activations, d_activations, bytes_activations, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_z, d_z, bytes_z, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_gradients, d_gradients, bytes_gradients, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_gradients_weights, d_gradients_weights, bytes_gradients_weights, cudaMemcpyDeviceToHost);

	// Free our memory
	cudaFree(d_biases);
	cudaFree(d_weights);
	cudaFree(d_activations);
	cudaFree(d_z);
	cudaFree(d_shape);
	cudaFree(d_target);
	cudaFree(d_activation_offset_table);
	cudaFree(d_wm_offset_table);
	cudaFree(d_gradients);
	cudaFree(d_gradient_offset_table);
	cudaFree(d_gradients_weights);

	int z_offset = 0;
	for (int shape_index = 1; shape_index < shape_length; shape_index++)
	{
		std::cout << "Z Values " << shape_index << ". hidden layer" << std::endl;
		
		for (int neuron_nr = 0; neuron_nr < shape[shape_index]; neuron_nr++)
		{
			std::cout << "[";
			for (int input = 0; input < NR_INPUTS; input++)
			{
				std::cout << host_z[z_offset + input * shape[shape_index] + neuron_nr] << ", ";
			}
			std::cout << "]" << std::endl;
			
		}
		z_offset += shape[shape_index] * NR_INPUTS;
	}

	int activations_offset = shape[0] * NR_INPUTS; // Skip input values	
	for (int shape_index = 1; shape_index < shape_length; shape_index++)
	{
		std::cout << "Activations " << shape_index << ". hidden layer" << std::endl;

		for (int neuron_nr = 0; neuron_nr < shape[shape_index]; neuron_nr++)
		{
			std::cout << "[";
			for (int input = 0; input < NR_INPUTS; input++)
			{
				std::cout << host_activations[activations_offset + input * shape[shape_index] + neuron_nr] << ", ";
			}
			std::cout << "]" << std::endl;
		}
		activations_offset += shape[shape_index] * NR_INPUTS;
	}

	int gradient_offset = 0;
	for (int shape_index = 1; shape_index < shape_length; shape_index++)
	{
		std::cout << "Gradients (Biases): " << shape_index << " . layer" << std::endl;
		for (int neuron_nr = 0; neuron_nr < shape[shape_index]; neuron_nr++)
		{
			std::cout << "[";
			for (int input = 0; input < NR_INPUTS; input++)
			{
				std::cout << host_gradients[gradient_offset + input * shape[shape_index] + neuron_nr] << ", ";
			}
			std::cout << "]" << std::endl;
		}
		gradient_offset += shape[shape_index] * NR_INPUTS;
		
	}

	int gradient_weight_offset = 0;
	for (int shape_index = 0; shape_index < shape_length - 1; shape_index++)
	{
		std::cout << "Gradients Layer (Weights): " << shape_index << " :" << std::endl;
		for (int input = 0; input < NR_INPUTS; input++)
		{
			std::cout << "Input " << input << ": " << std::endl;
			std::cout << "[";
			for (int next_layer_neuron = 0; next_layer_neuron < shape[shape_index + 1]; next_layer_neuron++)
			{
				std::cout << "[";
				for (int weight_counter = 0; weight_counter < shape[shape_index]; weight_counter++)
				{
					std::cout << host_gradients_weights[gradient_weight_offset + next_layer_neuron * shape[shape_index] + weight_counter] << ", ";
				}
				std::cout << "]" << std::endl;
			}
			gradient_weight_offset += shape[shape_index] * shape[shape_index + 1];
		}		
		std::cout << "]";		
	}

	getchar();


	return 0;
} 
