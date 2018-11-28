using System;
using static System.Console;
using System.Collections.Generic;
using System.Linq;

namespace Perceptron
{
    delegate double Function(double x);

    class Neuron
    {
        public Function ActivationFunction;
        public Function ActivationDerivative;
        public double[] Weights;

        static Random rnd = new Random();

        /// <summary>
        /// Initialising neuron with specific type
        /// </summary>
        /// <param name="nType"></param>
        /// <param name="numberOfInputs"></param>
        public Neuron(Function activationFunction, Function activationDerivative, int numberOfInputs, double maxWeight = 1)
        {
            int i;
            double[] weights = new double[++numberOfInputs];
            
            weights[0] = 1; //Bias

            for (i = 1; i < numberOfInputs; i++)
            {
                weights[i] = rnd.NextDouble() * maxWeight;
            }

            Weights = weights;
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

            if (inputs.Length != Weights.Length) throw new Exception("Length of input vector is not equal to length of weight vector.");

            for (i = 0; i < Weights.Length; i++)
            {
                summ += Weights[i] * inputs[i];
            }

            return ActivationFunction(summ);
        }

        /// <summary>
        /// Inserts extra space into 1-st element array
        /// </summary>
        /// <param name="inputArray"></param>
        /// <returns></returns>
        private double[] AddOne(double[] inputArray)
        {
            List<double> list = inputArray.ToList();

            list.Insert(0, 1);

            return list.ToArray();
        }
    }
}
