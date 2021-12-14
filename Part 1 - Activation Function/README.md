## 1. CPU Buffers

The first thing we are going to do is to create all the needed buffers. 

1. We need one buffer on the CPU for the z values 
2. And one for the activation values. 

We will call this host_z_values and host_activations. In the cuda world it’s convention to designate data which is stored on **cpu** accessible memory with **host**. And data which is stored on **GPU** accessible memory with **device**. 

```c
int main()
{
  const int arraySize = 5;

  // Initializing arrays on the CPU
  float host_z_values[arraySize] = { 1., 2., 3., 4., 5. };
  float host_activations[arraySize] = { 0 };
}
```


## 2. Request GPU Memory

First we calculate the numbers of bytes needed because we are going to need this for the cudaMalloc call.

```c
// Caclulating the number of bytes required to store the arrays.
const size_t bytes_z_values = arraySize * sizeof(float);
const size_t bytes_activations = arraySize * sizeof(float);
```

Additionally we need two pointers which will point to the GPU memory after the cuda malloc call.

```c
// Float pointers which will contain the address of the arrays on the GPU.
float *device_z_values, *device_activations;
```

Now we can call cudaMalloc, cudaMalloc expects two arguments, the first is the address of a pointer, the second is the number of bytes to allocate. cudaMalloc will change the value of the pointer so it points to free GPU memory with as much memory as we requested.
```c
// Allocate memory on the GPU
cudaMalloc(&device_z_values, bytes_z_values);
cudaMalloc(&device_activations, bytes_activations);
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
// Now that we have allocated memory space and the location is
// stored in our pointer we can transfer the values from the CPU to the GPU.
cudaMemcpy(device_z_values, host_z_values, bytes_z_values, cudaMemcpyHostToDevice);
```



## 4. Coding the Kernel
The goal is to compute the activation values of the given z array. 
The activation function we are going to use for that is the **sigmoid** function. 

For the compiler to know that we need a GPU function instead of a regular one we need to prepend the function definition with **__global__**.  
One essential thing we are provided with is the threadId. Which gives us the power to coordinate the work between threads. 
We will use the threadId to decide which activation value each thread will compute.

Each thread get’s one z value and will write the activation value into the activation_matrix buffer.
The formula on the right is just the sigmoid formula. 

```c
__global__ void sigmoidActivation(float *z_matrix, float *activation_matrix)
{   int index = threadIdx.x;
    activation_matrix[index] = 1.0 / (1.0 + exp(-z_matrix[index]));
}
```

## 5. Launching the Kernel

Alright that’s all the code we need for the kernel. We now have a kernel which can compute all the activation values at once. The only thing left to do now is to call the kernel. This can be done via the triple chevron launch syntax. 

```c
// Call the kernel which calculates the activations.
// 1 = Number of Blocks
// arraySize is the number of threads
sigmoidActivation << <1, arraySize >> > (device_z_values, device_activations);
```
For now we just need 5 threads(given by arraySize), later we will see how to scale things up, with more threads.

## 6. Transfering computed values back to CPU

While we now computed the activation values we still can't access them because they are still only available on GPU accessible memory.
In order to be able to print them we need to transfer them to the CPU. 
// Copy the results back to the CPU
cudaMemcpy(host_activations, device_activations, bytes_z_values, cudaMemcpyDeviceToHost);

## 7. Print the results
```c
printf("sigmoid({1,2,3,4,5}) = {%f,%f, %f,%f,%f}\n", host_activations[0], host_activations[1], host_activations[2], host_activations[3], host_activations[4]);
getchar();
```

Which should print the following:
```
sigmoid({1,2,3,4,5}) = {0.731059,0.880797, 0.952574, 0.982014, 0.993307}
```

[Full code](kernel.cu)
