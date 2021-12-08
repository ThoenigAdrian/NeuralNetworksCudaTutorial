
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>


__global__ void sigmoidActivation(float *z_matrix, float *activation_matrix)
{
    int index = threadIdx.x;
    activation_matrix[index] = 1.0 / (1.0 + exp(-z_matrix[index]));
}

int main()
{
    const int arraySize = 5;

	// Initializing arrays on the CPU
	float host_z_values[arraySize] = { 1., 2., 3., 4., 5. };
	float host_activations[arraySize] = { 0 };

	// Caclulating the number of bytes required to store the arrays.
	const size_t bytes_z_values = arraySize * sizeof(float);
	const size_t bytes_activations = arraySize * sizeof(float);

	// Float pointers which will contain the address of the arrays on the GPU.
	float *device_z_values, *device_activations;

	// Allocate memory on the GPU
	cudaMalloc(&device_z_values, bytes_z_values);
	cudaMalloc(&device_activations, bytes_activations);

	// Now that we have allocated memory space and the location is stored in our pointer we can transfer the values from the CPU to the GPU.
	cudaMemcpy(device_z_values, host_z_values, bytes_z_values, cudaMemcpyHostToDevice);


	// Call the kernel which calculates the activations.
	// 1 = Number of Blocks
	// arraySize is the number of threads
	sigmoidActivation << <1, arraySize >> > (device_z_values, device_activations);


	// Copy the results back to the CPU
	cudaMemcpy(host_activations, device_activations, bytes_z_values, cudaMemcpyDeviceToHost);
	
        printf("sigmoid({1,2,3,4,5}) = {%f,%f, %f,%f,%f}\n", host_activations[0], host_activations[1], host_activations[2], host_activations[3], host_activations[4]);
	getchar();

        return 0;
}
