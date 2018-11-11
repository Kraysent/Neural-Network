using static System.Console;
using static System.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Perceptron
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
            double learningRate = 0.4;
            Network neuralNetwork;

            Write("Enter number of inputs to the network: ");
            numberOfParameters = int.Parse(ReadLine());
            WriteLine("Creating network...");

            neuralNetwork = new Network(numberOfParameters, learningRate);
            neuralNetwork.AddLayer(numberOfParameters, Sigma, SigmaDerivative);
            neuralNetwork.AddLayer(1, Sigma, SigmaDerivative);
            WriteLine("Network created!");

            //-----------------Testing-----------------------//

            int numOfTests;
            double[] output;
            List<double[]> input = new List<double[]>();
            double[][] rightAnswers;
            bool networkResult;

            WriteLine();
            Write("Enter number of tests: ");
            numOfTests = int.Parse(ReadLine());
            WriteLine();
            WriteLine("Enter {0} tests with {1} parametres in each: ", numOfTests, numberOfParameters);

            for (i = 0; i < numOfTests; i++)
            {
                Write("Test {0}: ", i);
                input.Add(ReadLine().Split(' ').Select(x => double.Parse(x)).ToArray());
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

            WriteLine();

            WriteLine("Enter right answers: ");
            rightAnswers = new double[numOfTests][];

            for (i = 0; i < numOfTests; i++)
            {
                Write("For test {0}: ", i);
                rightAnswers[i] = ReadLine().Split(' ').Select(x => double.Parse(x)).ToArray();
            }
            
            WriteLine();
            WriteLine("Training...");
            networkResult = neuralNetwork.TrainUntilConvergence(input.ToArray(), rightAnswers, (int)1e6, 1e-4);
            WriteLine("Network {0}", (networkResult == true) ? "trained!" : "was not converged!");
            WriteLine();
            WriteLine("Network answers after training: ");

            for (i = 0; i < numOfTests; i++)
            {
                output = neuralNetwork.ForwardPass(input[i]);
                Write("For test {0}: ", i);

                for (j = 0; j < output.Length; j++) Write("{0} ", output[j]);

                WriteLine();
            }

            WriteLine();
            
            while (true)
            {
                Write("Ask something: ");
                output = neuralNetwork.ForwardPass(ReadLine().Split(' ').Select(x => double.Parse(x)).ToArray());
                Write("Network answer: ");

                for (i = 0; i < output.Length; i++)
                {
                    Write("{0} ", output[i]);
                }

                WriteLine();
                WriteLine();
            }
        }
    }
}
