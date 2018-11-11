using System;
using static System.Console;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkV2
{
    delegate double Function(double x);

    class Neuron
    {
        public Function ActivationFunction;
        public Function ActivationDerivative;
        public double[] InputWeights;

        /// <summary>
        /// Initialising neuron with specific type
        /// </summary>
        /// <param name="nType"></param>
        /// <param name="numberOfInputs"></param>
        public Neuron(Function activationFunction, Function activationDerivative, int numberOfInputs)
        {
            int i;
            double[] weights = new double[++numberOfInputs];
            Random rnd = new Random();

            weights[0] = 1; //Bias

            for (i = 1; i < numberOfInputs; i++)
            {
                weights[i] = (double)rnd.Next(0, 10) / 10;
            }

            InputWeights = weights;
            ActivationFunction = activationFunction;
            ActivationDerivative = activationDerivative;
        }

        /// <summary>
        /// One activation for current neuron. Inputs must be WITHOUT included bias.
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double ForwardPass(double[] inputs)
        {
            int i;
            double summ = 0;

            inputs = AddOne(inputs);

            if (inputs.Length != InputWeights.Length) throw new Exception("Length of input vector is not equal to length of weight vector.");

            for (i = 0; i < InputWeights.Length; i++)
            {
                summ += InputWeights[i] * inputs[i];
            }

            return ActivationFunction(summ);
        }

        private double[] AddOne(double[] inputArray)
        {
            List<double> list = inputArray.ToList();

            list.Insert(0, 1);

            return list.ToArray();
        }
    }
}
