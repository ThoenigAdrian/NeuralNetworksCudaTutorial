#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include "kernel.h"
#include "json.hpp"
#include <fstream>
#include <numeric>
#include <algorithm>
#include "matrix_print.hpp"


__global__ void feed_forward_and_backprop(float *z_matrix, float* activation_matrix, float* weight_matrix, float* biases, int* shape, int shape_length, float* outputs, float* error, float* error_weights,
	int* activation_offset_table, int* bias_offset_table, int* z_offset_table, int* wm_offset_table)
{
	// Feed forward of an entire neural network for one single input
	// We go trhough each layer step by step and calculate the neurons activations
	// After each layer we synchronize the threads 
	// But now with multiple inputs
	// blockDim.x is the same as number_of_inputs
	// blockDim.y is the maximum number of neurons per layer of the entire network example_shape = [7,9,3] => blockDim.y = 9
	// blockDim.y isn't used in the code but it's required for launch enough threads
	// threadIdx.x is the current input nr
	// threadIdx.y is the current neuron in the l+1th layer (also called the outgoing neuron)
	// Now we also use backpropagation there might be some changes as how we use the threads especially for BP4

	int output_neurons = shape[shape_length - 1];
	int *error_offset_table = z_offset_table;
	int input_offset = 0;
	int input_offset_next = 0;

	for (int layer = 0; layer < shape_length - 1; layer++)
	{
		if (threadIdx.y < shape[layer+1]) // for every outgoing neuron one thread
		{
			input_offset = shape[layer] * threadIdx.x;
			input_offset_next = shape[layer+1] * threadIdx.x;

			// w*x
			for (int incoming_neuron = 0; incoming_neuron < shape[layer]; incoming_neuron++)
			{
				z_matrix[z_offset_table[layer] + input_offset_next + threadIdx.y] += weight_matrix[wm_offset_table[layer] + (shape[layer]-1)*incoming_neuron + threadIdx.y] * activation_matrix[activation_offset_table[layer] + input_offset + incoming_neuron];
			}
		
			// w*x + b
			z_matrix[z_offset_table[layer] + input_offset_next + threadIdx.y] += biases[bias_offset_table[layer] + threadIdx.y];

			// sig(w*x + b)
			activation_matrix[activation_offset_table[layer+1] + input_offset_next + threadIdx.y] = 1.0 / (1.0 + exp(-z_matrix[z_offset_table[layer] + input_offset_next + threadIdx.y]));

		}
		__syncthreads();
	}
	if (threadIdx.y < shape[shape_length - 1])
	{
		// output doesn't need the offset as it's just the final layer
		// error and activation matrix need the offset for the last layer because they contain information about all the layers
		// layer_offset is still set from the forward propagation to exactly the last layer
		// but the error matrix has no values for the input as it doesnt make sense so we substract the first layer offset
		error[error_offset_table[shape_length-2] + threadIdx.x * output_neurons + threadIdx.y] = activation_matrix[activation_offset_table[shape_length-1] + threadIdx.x * output_neurons + threadIdx.y] - outputs[threadIdx.x * output_neurons + threadIdx.y];
	}
	for (int layer = shape_length - 2; layer > 0; layer--)
	{
		// BP 2 + 3
		// threadIdx.y is the neuron in the current layer we need to sum up error1 * w11 + error2 * w21 + error3 * w31....
		// and we need to do this for all inputs threadIdx.x
		if (threadIdx.y < shape[layer])
		{
			for (int higher_layer_neuron_nr = 0; higher_layer_neuron_nr < shape[layer + 1]; higher_layer_neuron_nr++)
			{
				error[error_offset_table[layer-1] + threadIdx.y + threadIdx.x * shape[layer]] += error[error_offset_table[layer] + higher_layer_neuron_nr + threadIdx.x * shape[layer+1]] * 
													weight_matrix[wm_offset_table[layer] + threadIdx.y * shape[layer+1] + higher_layer_neuron_nr];
			}
			float sig_strich = activation_matrix[activation_offset_table[layer] + threadIdx.y + threadIdx.x * shape[layer]] * (1 - activation_matrix[activation_offset_table[layer] + threadIdx.y + threadIdx.x * shape[layer]]);
			error[error_offset_table[layer-1] + threadIdx.y + threadIdx.x * shape[layer]] *= sig_strich;
		}
		__syncthreads();
	}


	// BP4 
	for (int layer = 0; layer < shape_length - 1; layer++)
	{
		int nr_of_threads = shape[layer] * shape[layer + 1];
		// for each weight one thread
		if (threadIdx.y < shape[layer] * shape[layer + 1])
		{
			// I know this is yet not efficient for many inputs . Problems is it's not that easy because we need to access the same memory address when summing
			// There is an summing algorithm which is more efficient but maybe later
			for (int input_nr = 0; input_nr < blockDim.x; input_nr++)
			{
				error_weights[wm_offset_table[layer] + threadIdx.y] += 
					activation_matrix[activation_offset_table[layer] + input_nr*shape[layer] + threadIdx.y / shape[layer + 1]] *
					 error[error_offset_table[layer] + input_nr*shape[layer+1] + threadIdx.y % shape[layer + 1]];
			}
			
		}
		__syncthreads();
	}

}


void calculate_offsets(int* activation_offset_table, int *bias_offset_table, int *z_offset_table, int *weight_matrix_offset_table, int* shape, int shape_length, int nr_of_inputs)
{
	// Goal of this to get the correct offset for a certain layer by offset_table[layer]
	activation_offset_table[0] = 0;
	bias_offset_table[0] = 0;
	z_offset_table[0] = 0;
	weight_matrix_offset_table[0] = 0;
	for (int i = 1; i < shape_length; i++)
	{
		activation_offset_table[i] = activation_offset_table[i - 1] + shape[i - 1] * nr_of_inputs;
	}
	for (int i = 1; i < shape_length - 1; i++)
	{
		bias_offset_table[i] = bias_offset_table[i - 1] + shape[i];
		z_offset_table[i] = bias_offset_table[i] * nr_of_inputs;
		weight_matrix_offset_table[i] = weight_matrix_offset_table[i - 1] + shape[i] * shape[i - 1];
	}
}



int main()
{
	using json = nlohmann::json;
	std::ifstream i("D:\\Privat\\Programmieren\\CudaNeuralNets\\forward_pass4\\fw_pass4\\network.json");
	json network_init_json = json::parse(i);
	int NUM_NEURONS_TOTAL = accumulate(network_init_json["shape"].begin(), network_init_json["shape"].end(), 0);
	int NUM_LAYERS_TOTAL = (int)network_init_json["shape"].size();
	int *SHAPE = (int*)malloc(network_init_json["shape"].size() * sizeof(int));
	int *SHAPE_WEIGHTS = (int*)malloc((network_init_json["shape"].size() - 1) * sizeof(int));
	std::copy(network_init_json["shape"].begin(), network_init_json["shape"].end(), SHAPE);

	if (network_init_json["input"].size() % SHAPE[0] != 0)
	{
		std::cout << "Shape of input doesn't match should be a multiple of " << SHAPE[0] << std::endl;
		return 0;
	}
	int number_of_inputs = (int)network_init_json["input"].size() / SHAPE[0];
		
	// Initilaize weight sizes from json
	int weight_sizes = 0;
	for (int i = 0; i < network_init_json["shape"].size() - 1; i++)
	{
		weight_sizes += (int)network_init_json["shape"][i] * (int)network_init_json["shape"][i + 1];
		SHAPE_WEIGHTS[i] = (int)network_init_json["shape"][i] * (int)network_init_json["shape"][i + 1];
	}
	
	// Initilaize bias sizes from json
	const int bias_sizes = NUM_NEURONS_TOTAL - SHAPE[0];

	// Calculate all the needed sizes
	const size_t bytes_biases = bias_sizes * sizeof(float);
	const size_t bytes_z = bias_sizes * sizeof(float) * number_of_inputs;
	const size_t bytes_error = bias_sizes * sizeof(float) * number_of_inputs;
	const size_t bytes_error_weights = weight_sizes * sizeof(float);
	const size_t bytes_weights = weight_sizes * sizeof(float);
	const size_t bytes_activations = (NUM_NEURONS_TOTAL) * sizeof(float) * number_of_inputs;
	const size_t bytes_shape = NUM_LAYERS_TOTAL * sizeof(int);
	const size_t bytes_output = (int)network_init_json["output"].size() * sizeof(float);
	const size_t bytes_activation_offset = NUM_LAYERS_TOTAL * sizeof(int);
	const size_t bytes_biass_offset = (NUM_LAYERS_TOTAL - 1) * sizeof(int);
	const size_t bytes_z_offset = bytes_biass_offset;
	const size_t bytes_wm_offset = bytes_biass_offset;

	// Lookup tables for the offsets 
	int *activation_offset_table = (int*)malloc(bytes_activation_offset);
	int *bias_offset_table = (int*)malloc(bytes_biass_offset);
	int *z_offset_table = (int*)malloc(bytes_z_offset);
	int *weight_matrix_offset_table = (int*)malloc(bytes_wm_offset);
	calculate_offsets(activation_offset_table, bias_offset_table, z_offset_table, weight_matrix_offset_table, SHAPE, NUM_LAYERS_TOTAL, number_of_inputs);

	/*---------------- Fill weights ----------------------*/
	float *host_weights = (float*) malloc(bytes_weights);
	std::copy(network_init_json["weights"].begin(), network_init_json["weights"].end(), host_weights);
	print_weight_matrix_numpy_style(host_weights, SHAPE, NUM_LAYERS_TOTAL);
	
	/*---------------- Fill biases ----------------------*/
	float *host_biases = (float*)malloc(bytes_biases);
	std::copy(network_init_json["biases"].begin(), network_init_json["biases"].end(), host_biases);
	print_bias_matrix_numpy_style(host_biases, SHAPE, NUM_LAYERS_TOTAL);
	/*---------------- COPY BIASES INTO ONE BIG ARRAY----------------------*/
	
	// Copy Input into activation matrix	
	float *host_activations = (float*)malloc(bytes_activations);
	std::copy(network_init_json["input"].begin(), network_init_json["input"].end(), host_activations);

	// Copy Output into output matrix
	float *host_outputs = (float*)malloc(bytes_output);
	std::copy(network_init_json["output"].begin(), network_init_json["output"].end(), host_outputs);
	
	// Initialize z Matrix
	float *host_z  = (float*)malloc(bytes_z);
	memset(host_z, 0, bytes_z);

	float *host_error = (float*)malloc(bytes_error);
	memset(host_error, 0, bytes_error);

	float *host_error_weights = (float*)malloc(bytes_error_weights);
	memset(host_error_weights, 0, bytes_error_weights);

	// Allocate GPU device  memory
	float *d_biases, *d_weights, *d_activations, *d_z, *d_error, *d_error_weights, *d_output;
	int *d_shape, *d_activation_offset_table, *d_bias_offset_table, *d_z_offset_table, *d_weight_matrix_offset_table;
	cudaMalloc(&d_error, bytes_error);
	cudaMalloc(&d_output, bytes_output);
	cudaMalloc(&d_biases, bytes_biases);
	cudaMalloc(&d_weights, bytes_weights);
	cudaMalloc(&d_error_weights, bytes_error_weights);
	cudaMalloc(&d_activations, bytes_activations);
	cudaMalloc(&d_z, bytes_z);
	cudaMalloc(&d_shape, bytes_shape);
	cudaMalloc(&d_activation_offset_table, bytes_activation_offset);
	cudaMalloc(&d_bias_offset_table, bytes_biass_offset);
	cudaMalloc(&d_z_offset_table, bytes_z_offset);
	cudaMalloc(&d_weight_matrix_offset_table, bytes_wm_offset);

	// Copy data to GPU
	cudaMemcpy(d_error_weights, host_error_weights, bytes_error_weights, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, host_outputs, bytes_output, cudaMemcpyHostToDevice);
	cudaMemcpy(d_error, host_error, bytes_error, cudaMemcpyHostToDevice);
	cudaMemcpy(d_biases, host_biases, bytes_biases, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, host_weights, bytes_weights, cudaMemcpyHostToDevice);
	cudaMemcpy(d_activations, host_activations, bytes_activations, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, host_z, bytes_z, cudaMemcpyHostToDevice);
	cudaMemcpy(d_shape, SHAPE, bytes_shape, cudaMemcpyHostToDevice);

	cudaMemcpy(d_activation_offset_table, activation_offset_table, bytes_activation_offset, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias_offset_table, bias_offset_table, bytes_biass_offset, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z_offset_table, z_offset_table, bytes_z_offset, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight_matrix_offset_table, weight_matrix_offset_table, bytes_wm_offset, cudaMemcpyHostToDevice);
	
	// Prepare call parameters threadblock size and gride size
	int max_threads_gtx1070ti = 1024;
	int x_dim = number_of_inputs;
	// preivious version but changed for BP4 int y_dim = *std::max_element(SHAPE, SHAPE + network_init_json["shape"].size() - 1); 
	int y_dim = *std::max_element(SHAPE_WEIGHTS, SHAPE_WEIGHTS + network_init_json["shape"].size() - 2);
	dim3 dimBlock(x_dim, y_dim, 1);
	int num_blocks = ((x_dim * y_dim) / max_threads_gtx1070ti + 1);
	
	if (num_blocks > 1)
	{
		std::cout << "Can't handle this yet";
		return 0;
	}
	dim3 dimGrid(num_blocks, 1, 1);

	// Call cuda feed forward function for multiple inputs

	feed_forward_and_backprop << <dimGrid, dimBlock >> > (d_z, d_activations, d_weights, d_biases, d_shape, NUM_LAYERS_TOTAL, d_output, d_error, d_error_weights, d_activation_offset_table, d_bias_offset_table, d_z_offset_table, d_weight_matrix_offset_table);

	// Copy data from GPU to RAM
	cudaMemcpy(host_activations, d_activations, bytes_activations, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_weights, d_weights, bytes_weights, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_biases, d_biases, bytes_biases, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_z, d_z, bytes_z, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_error, d_error, bytes_error, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_error_weights, d_error_weights, bytes_error_weights, cudaMemcpyDeviceToHost);
	
	// Free our memory
	cudaFree(d_biases);
	cudaFree(d_weights);
	cudaFree(d_activations);
	cudaFree(d_shape);
	cudaFree(d_z);
	cudaFree(d_error);
	cudaFree(d_error_weights);

	print_activations(host_activations, number_of_inputs, SHAPE, NUM_LAYERS_TOTAL);
	print_error(host_error, number_of_inputs, SHAPE, NUM_LAYERS_TOTAL);
	print_weight_matrix_numpy_style(host_error_weights, SHAPE, NUM_LAYERS_TOTAL);

	getchar();

	return 0;
}