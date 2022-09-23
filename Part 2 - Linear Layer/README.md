## Linear Layer

Last time we assumed a given **z value** in order to be able to compute the activation function.
This time we are actually going to compute the **z value** , also known as weighted sum.
In neural networks the thing which computes the weighted sum is the **Linear Layer**.

The first thing we need before we start is a plan on how we should exploit the many threads we can use.
So we need to come up with a scheme on how to distribute the work among the threads.

We are going to use the follwoing idea:

One thread takes care of one neuron. So if we have 3 output neurons we are going to use 3 threads.
Each of this threads uses the weights which belongs to it's assigned output neuron and calculates the weighted sum and then calculates the activation value.

![](threadtimeline.png)

```
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
	
}```
