using static System.Console;
using static System.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkV2
{
    class Program
    {
        static double Sigma(double x) => 1 / (1 + Exp(-x));
        static double SigmaDerivative(double x) => Sigma(x) * (1 - Sigma(x));
        static double Pass(double x) => x;
        
        static void Main(string[] args)
        {
            //-----------------Creating network-----------------//

            int numberOfParameters = 2, i, j;
            double learningRate = 0.1;
            Network neuralNetwork = new Network(numberOfParameters, learningRate);

            neuralNetwork.AddLayer(2, Sigma, SigmaDerivative);
            neuralNetwork.AddLayer(1, Sigma, SigmaDerivative);

            //-----------------Testing-----------------------//

            int numOfTests;
            double[] output;
            List<double[]> input = new List<double[]>();

            Write("Enter number of tests: ");
            numOfTests = int.Parse(ReadLine());
            WriteLine();
            WriteLine("Enter {0} tests with {1} parametres in each: ", numOfTests, numberOfParameters);

            for (i = 0; i < numOfTests; i++)
            {
                Write("Test {0}: ", i);
                input.Add(ReadLine().Split(' ').Select(x => double.Parse(x)).ToArray());
                input[i] = AddingOne(input[i]);
            }

            WriteLine();
            WriteLine("Network answers: ");

            for (i = 0; i < numOfTests; i++)
            {
                output = neuralNetwork.ForwardPass(input[i]);
                Write("For test {0}: ", i);

                for (j = 0; j < output.Length; j++) Write("{0} ", output[j]);

                WriteLine(); 
            }
            
            ReadKey();
        }
        
        private static double[] AddingOne(double[] input)
        {
            List<double> arr = input.ToList();

            arr.Insert(0, 1);

            return arr.ToArray();
        }
    }
}
