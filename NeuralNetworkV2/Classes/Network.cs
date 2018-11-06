using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworkV2
{
    class Network
    {
        public int NumberOfInputs;
        public List<Neuron[]> Neurons;
        public double LearningRate;

        public int NumberOfLayers { get => Neurons.Count; }
        public int NumberInOutputLayer { get => Neurons[NumberOfLayers - 1].Length; }

        /// <summary>
        /// Initialises empty neural network
        /// </summary>
        /// <param name="numberOfInputs"></param>
        public Network(int numberOfInputs, double learningRate)
        {
            Neurons = new List<Neuron[]>();
            NumberOfInputs = numberOfInputs;
        }

        /// <summary>
        /// Adds new layer of neurons to network
        /// </summary>
        /// <param name="numberOfNeurons"></param>
        /// <param name="activation"></param>
        /// <param name="activationDerivative"></param>
        public void AddLayer(int numberOfNeurons, Function activation, Function activationDerivative)
        {
            int i, currLayer = Neurons.Count;

            Neurons.Add(new Neuron[numberOfNeurons]);
            
            for (i = 0; i < numberOfNeurons; i++)
            {
                Neurons[currLayer][i] = new Neuron(activation, activationDerivative, NumberOfInputs);
            }
        }

        /// <summary>
        /// Network answer for single test
        /// </summary>
        /// <param name="inputs"></param>
        /// <returns></returns>
        public double[] ForwardPass(double[] inputs)
        {
            int i, j;
            double[] outputVector = new double[inputs.Length];

            for (i = 0; i < Neurons.Count; i++)
            {
                outputVector = new double[Neurons[i].Length];

                for (j = 0; j < Neurons[i].Length; j++)
                {
                    outputVector[j] = Neurons[i][j].ForwardPass(inputs);
                }

                inputs = AddingOne(outputVector);
            }

            return outputVector;
        }

        public void TrainOnSingleExample(double[] example, double[] answer)
        {
            //-------Forward pass with saved answers for each neuron-------//

            int i, j;
            double[] inputVector = example;
            List<double[]> outputs = new List<double[]>();

            for (i = 0; i < Neurons.Count; i++)
            {
                outputs.Add(new double[Neurons[i].Length]);

                for (j = 0; j < Neurons[i].Length; j++)
                {
                    outputs[i][j] = Neurons[i][j].ForwardPass(inputVector);
                }

                inputVector = AddingOne(outputs[i]);
            }

            //-------Training-------//

            double[][] delta = new double[NumberInOutputLayer][];
            int k;

            for (i = NumberOfLayers - 1; i >= 0; i--)
            {
                delta[i] = new double[Neurons[i].Length];

                for (j = 0; j < Neurons[i].Length; j++)
                {
                    /*
                     * delta = SigmaDerivative(summatory of current neuron) * 
                     *                                                          * ScalarProduct(Output weights of curr neuron, deltas of prev layer) for NOT output neurons
                     *                                                          or
                     *                                                          * (T - O) for output neurons, T - target answer, O - real answer
                     */
                    delta[i][j] = outputs[i][j] * (1 - outputs[i][j]) * ((i == NumberOfLayers - 1) ? (answer[j] - outputs[i][j]) : ScalarProduct(GetOutputWeights(i, j), delta[i + 1])); 

                    for (k = 0; k < Neurons[i][j].InputWeights.Length; k++)
                    {
                        Neurons[i][j].InputWeights[k] = Neurons[i][j].InputWeights[k] + delta[i][j] * LearningRate * ((i == 0) ? example[k] : outputs[i - 1][k]);
                    }
                }
            }

            //-------Counting error-------//


        }

        /// <summary>
        /// Gets array of weights going out from current neuron
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="number"></param>
        /// <returns></returns>
        public double[] GetOutputWeights(int layer, int number)
        {
            int i, j;
            List<double> output = new List<double>();

            if (layer == (Neurons.Count - 1)) throw new Exception("Output layer has no output weights");
            
            for (i = 0; i < Neurons[layer].Count(); i++)
            {
                for (j = 0; j < Neurons[layer][i].InputWeights.Length; j++)
                {
                    if (j == number)
                    {
                        output.Add(Neurons[layer][i].InputWeights[j]);
                    }
                }
            }

            return output.ToArray();
        }

        private static double[] AddingOne(double[] input)
        {
            List<double> arr = input.ToList();

            arr.Insert(0, 1);

            return arr.ToArray();
        }

        private static double ScalarProduct(double[] vector1, double[] vector2)
        {
            int i;
            double sum = 0;

            if (vector1.Length != vector2.Length) throw new Exception("Vectors have not the same length");

            for (i = 0; i < vector1.Length; i++)
            {
                sum += vector2[i] * vector1[i];
            }

            return sum;
        }
    }
}
