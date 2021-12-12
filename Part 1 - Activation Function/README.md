INCOMPLETE!! I will show hot to implement neural networks in CUDA from Scratch. This means no libraries, not even the linear algebra library of NVIDA called CUBLAS.
If you want to follow along make sure you have a working CUDA development environment.
This is not an advanced CUDA tutorial we will only cover the parts of CUDA we need in order to program our neural network. CUDA Programming is a deep topic, from knowing about the GPU Hardware Architecture, the different memory types, good memory access patterns, the threading model, performance profiling, compatibility between different models of graphics card and many more things.  If you’re mainly interested in this type of things you should check out some of the tutorials I linked in the description.

I will show hot to implement neural networks in CUDA from Scratch. This means no libraries, not even the linear algebra library of NVIDA called CUBLAS.
If you want to follow along make sure you have a working CUDA development environment.
This is not an advanced CUDA tutorial we will only cover the parts of CUDA we need in order to program our neural network. CUDA Programming is a deep topic, from knowing about the GPU Hardware Architecture, the different memory types, good memory access patterns, the threading model, performance profiling, compatibility between different models of graphics card and many more things.  If you’re mainly interested in this type of things you should check out some of the tutorials I linked in the description.

##1 CPU Buffers

The first thing we are going to do is to create all the needed buffers. We need one buffer on the CPU for the z values and one for the activation values. 
We will call this host_z_values and host_activations. In the cuda world it’s convention to designate data which is stored on cpu accessible memory with host. And data which is stored on the GPU memory with device. 

##2 Request GPU Memory

First we calculate the numbers of bytes needed because we are going to need this for the cudaMalloc call.
Additionally we need two pointers which will point to the GPU memory after the cuda malloc call.
Now we can call cudaMalloc, cudaMalloc expects two arguments, the first is the address of a pointer, the second is the number of bytes to allocate. cudaMalloc will change the value of the pointer so it points to free GPU memory with as much memory as we requested.

##3 Transfer z Values from CPU to GPU

This can be done with cudaMemcpy. The function signature is as follows, destination buffer, source buffer, number of bytes, and the direction. 
For the direction there is: Host to Device and Device to Host.

##4 Coding the Kernel

