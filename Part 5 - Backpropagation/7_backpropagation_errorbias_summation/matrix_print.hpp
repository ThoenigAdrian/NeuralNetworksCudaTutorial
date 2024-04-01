#include <iostream>
#include <sstream>
#include <string>

inline void print_matrix_numpy_style(float* thematrix, unsigned int zeilen, unsigned int spalten, std::string name)
{
	/* zeilen starts at */
	std::cout << name << " = ";
	std::cout << "np.array([";
	for (unsigned int zeilennummer = 0; zeilennummer < zeilen; zeilennummer++)
	{
		std::cout << "[";
		for (unsigned int spaltennummer = 0; spaltennummer < spalten; spaltennummer++)
		{
			std::cout << thematrix[spaltennummer * zeilen + zeilennummer] << ", ";
			// std::cout << spaltennummer * zeilen + zeilennummer << ", ";
		}
		std::cout << "]";
		if (zeilennummer != zeilen - 1)
		{
			std::cout << ","; // Add a sepeartion commma but don't do it for the last element so there are no trailing commas
		}
		std::cout << std::endl;
	}
	std::cout << "])\n";

}

inline void print_error(float* host_error, int number_of_inputs, int* SHAPE, int NUM_LAYERS_TOTAL)
{
	std::stringstream ss;
	int offset = 0;
	for (int layer = 1; layer < NUM_LAYERS_TOTAL; layer++)
	{
		ss.str("");
		ss.clear();
		ss << "error" << layer;
		print_matrix_numpy_style(host_error + offset, SHAPE[layer], number_of_inputs, ss.str());
		offset += SHAPE[layer] * number_of_inputs;
	}
}

inline void print_activations(float* host_activations, int number_of_inputs, int* SHAPE, int NUM_LAYERS_TOTAL)
{
	std::stringstream ss;
	int offset = 0;
	for (int layer = 0; layer < NUM_LAYERS_TOTAL; layer++)
	{
		ss.str("");
		ss.clear();
		ss << "activation" << layer;
		print_matrix_numpy_style(host_activations + offset, SHAPE[layer], number_of_inputs, ss.str());
		offset += SHAPE[layer] * number_of_inputs;
	}
}


inline void print_weight_matrix_numpy_style(float *the_matrix, const int shape[], int length)
{
	int offset = 0;
	std::stringstream ss;
	for (int i = 0; i < length - 1; i++)
	{
		ss.str("");
		ss.clear();
		ss << "weight_matrix" << i;
		print_matrix_numpy_style(the_matrix + offset, shape[i + 1], shape[i], ss.str());
		offset += shape[i] * shape[i + 1];
	}
}

inline void print_bias_matrix_numpy_style(float *the_matrix, const int shape[], int shape_length)
{
	int offset = 0;
	std::stringstream ss;
	for (int i = 0; i < shape_length - 1; i++)
	{
		ss.str("");
		ss.clear();
		ss << "bias_layer" << i;
		print_matrix_numpy_style(the_matrix + offset, shape[i + 1], 1, ss.str());
		offset += shape[i + 1];
	}
}