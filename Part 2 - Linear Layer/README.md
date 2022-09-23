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
