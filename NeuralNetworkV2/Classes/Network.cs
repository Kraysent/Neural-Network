using System;
using static System.IO.File;
using static System.Math;
using static System.Console;
using System.Collections.Generic;
using System.Linq;

// Works only with sigma activation yet

namespace Perceptron
{
    class Network
    {
        public const char SEPARATOR = ';';

        public int NumberOfInputs;
        public List<Neuron[]> Neurons;
        public double LearningRate;

        public int NumberOfLayers { get => Neurons.Count; }
        public int NumberOfOutputs { get => Neurons[NumberOfLayers - 1].Length; }

        public static double Sigma(double x) => 1 / (1 + Exp(-x));
        public static double SigmaDerivative(double x) => Sigma(x) * (1 - Sigma(x));
        
        /// <summary>
        /// Initialises empty neural network
        /// </summary>
        /// <param name="numberOfInputs"></param>
        public Network(int numberOfInputs, double learningRate = 0.1)
        {
            Neurons = new List<Neuron[]>();
            NumberOfInputs = numberOfInputs;
            LearningRate = learningRate;
        }

        /// <summary>
        /// Adds new layer of neurons to network
        /// </summary>
        /// <param name="numberOfNeurons"></param>
        /// <param name="activation"></param>
        /// <param name="activationDerivative"></param>
        public void AddLayer(int numberOfNeurons)
        {
            int i, currLayer = Neurons.Count;

            Neurons.Add(new Neuron[numberOfNeurons]);
            
            for (i = 0; i < numberOfNeurons; i++)
            {
                Neurons[currLayer][i] = new Neuron(Sigma, SigmaDerivative, (NumberOfLayers == 1) ? NumberOfInputs : Neurons[currLayer - 1].Length);
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
            double[] outputVector;
            
            outputVector = new double[inputs.Length];

            for (i = 0; i < NumberOfLayers; i++)
            {
                outputVector = new double[Neurons[i].Length];
                
                for (j = 0; j < Neurons[i].Length; j++)
                {
                    outputVector[j] = Neurons[i][j].ForwardPass(inputs);
                }

                inputs = outputVector;
            }

            return outputVector;
        }

        /// <summary>
        /// Changes weight of each neuron and returns post-error for single example
        /// </summary>
        /// <param name="example"></param>
        /// <param name="answer"></param>
        /// <returns></returns>
        public double TrainOnSingleExample(double[] example, double[] answer)
        {
            if (example.Length != NumberOfInputs) throw new Exception("Example vector has not the same length as number of inputs to network");
            if (answer.Length != NumberOfOutputs) throw new Exception("Answers vector has not the same length as number of outputs from network");

            //-------Forward pass with saved answers for each neuron-------//

            int i, j;
            double[] inputVector = example;
            List<double[]> outputs = new List<double[]>();
            
            for (i = 0; i < NumberOfLayers; i++)
            {
                outputs.Add(new double[Neurons[i].Length]);

                for (j = 0; j < Neurons[i].Length; j++)
                {
                    outputs[i][j] = Neurons[i][j].ForwardPass(inputVector);
                }
                
                inputVector = outputs[i];
            }

            //-------Training-------//

            double[][] delta = new double[NumberOfLayers][];
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
                     *                                                          * (T - O), where T - target answer, O - real answer, for output neurons
                     */
                    delta[i][j] = outputs[i][j] * (1 - outputs[i][j]) * ((i == NumberOfLayers - 1) ? (answer[j] - outputs[i][j]) : ScalarProduct(GetOutputWeights(i, j), delta[i + 1]));
                }
            }

            for (i = 0; i < NumberOfLayers; i++)
            {
                for (j = 0; j < Neurons[i].Length; j++)
                {
                    for (k = 0; k < Neurons[i][j].Weights.Length; k++)
                    {
                        Neurons[i][j].Weights[k] += delta[i][j] * LearningRate * ((k == 0) ? 1 : ((i == 0) ? example[k - 1] : outputs[i - 1][k - 1]));
                    }
                }
            }

            //-------Counting error-------//
            
            double error = 0;
            double[] newAnswer = ForwardPass(example);

            for (i = 0; i < newAnswer.Length; i++)
            {
                error += Pow(newAnswer[i] - answer[i], 2);
            }

            return error;
        }

        /// <summary>
        /// Trains network on input matrix of data. Returns true, if network converged, false if not
        /// </summary>
        /// <param name="inputMatrix"></param>
        /// <param name="rightAnswers"></param>
        /// <param name="maxEpoches"></param>
        /// <param name="eps"></param>
        /// <returns></returns>
        public bool TrainUntilConvergence(double[][] inputMatrix, double[][] rightAnswers, int maxEpoches = (int)1e6, double eps = 1e-4)
        {
            if (inputMatrix.Length != rightAnswers.Length) throw new Exception("Number of example tests is not equal to number of right answers tests");

            int i, j;
            double prevErr = 0, currError = 0;

            for (i = 0; i < maxEpoches; i++)
            {
                currError = 0;

                for (j = 0; j < inputMatrix.Length; j++)
                {
                    //WriteLine(" --- Example {0}, epoch {1}", j, i);
                    currError += TrainOnSingleExample(inputMatrix[j], rightAnswers[j]);
                    //WriteLine("Current error: {0}, previous error: {1}", currError, prevErr);
                }

                /*WriteLine("Network answers on epoch {0}: ", i);

                for (j = 0; j < inputMatrix.Length; j++)
                {
                    double[] output = ForwardPass(inputMatrix[j]);
                    Write("For test {0}: ", j);
                    
                    for (int k = 0; k < output.Length; k++) Write("{0} ", output[k]);

                    WriteLine();
                }*/

                if (Abs(currError) < eps) return true;

                prevErr = currError;
            }

            return false;
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

            layer++; //Output weights are on the next layer
            number++; //Because bias shifts all weights (and they do not fit numbering of neurons)

            if (layer == (NumberOfLayers)) throw new Exception("Output layer has no output weights");
            
            for (i = 0; i < Neurons[layer].Count(); i++)
            {
                for (j = 0; j < Neurons[layer][i].Weights.Length; j++)
                {
                    if (j == number)
                    {
                        output.Add(Neurons[layer][i].Weights[j]);
                    }
                }
            }

            return output.ToArray();
        }
        
        /// <summary>
        /// Uploads network to file
        /// </summary>
        /// <param name="outputDirectory"></param>
        public void Upload(string outputFileName)
        {
            List<string> outputList = new List<string>();
            string currString;
            int i, j, k;
            
            currString = "";

            for (i = 0; i < NumberOfLayers; i++)
            {
                currString += Neurons[i].Length + ((i != NumberOfLayers - 1) ? SEPARATOR.ToString() : "");
            }

            outputList.Add(currString);

            for (i = 0; i < NumberOfLayers; i++)
            {
                for (j = 0; j < Neurons[i].Length; j++)
                {
                    currString = "";

                    for (k = 0; k < Neurons[i][j].Weights.Length; k++)
                    {
                        currString += Neurons[i][j].Weights[k] + ((k != Neurons[i][j].Weights.Length - 1) ? SEPARATOR.ToString() : "");
                    }

                    outputList.Add(currString);
                }
            }

            WriteAllLines(outputFileName, outputList);
        }

        /// <summary>
        /// Downloads network from file
        /// </summary>
        /// <param name="fileName"></param>
        public void Download(string fileName)
        {
            try
            {
                double[][] contents = ReadAllLines(fileName).Select(x => x.Split(SEPARATOR).Select(y => double.Parse(y)).ToArray()).ToArray();
                int i, j, k = 0;

                Clear();

                for (i = 0; i < contents[0].Length; i++)
                {
                    AddLayer((int)contents[0][i]);
                }

                for (i = 0; i < contents[0].Length; i++)
                {
                    for (j = 0; j < contents[0][i]; j++)
                    {
                        k++;
                        Neurons[i][j].Weights = contents[k];
                    }
                }
            }
            catch { throw new Exception("File has wrong format"); }
        }

        /// <summary>
        /// Clears current network
        /// </summary>
        public void Clear()
        {
            Neurons.Clear();
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
