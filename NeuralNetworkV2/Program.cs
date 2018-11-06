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

            int numberOfParameters = 2, i;
            double learningRate = 0.1;
            Network neuralNetwork = new Network(numberOfParameters, learningRate);

            neuralNetwork.AddLayer(2, Sigma, SigmaDerivative);
            neuralNetwork.AddLayer(1, Sigma, SigmaDerivative);

            //-----------------Creating network-----------------//

            //-----------------Testing-----------------------//

            double[] input, output;

            Write("Enter {0} numbers: ", numberOfParameters);
            input = AddingOne(ReadLine().Split(' ').Select(x => double.Parse(x)).ToArray());
            output = neuralNetwork.ForwardPass(input.ToArray());

            Write("Network answer: ");

            for (i = 0; i < output.Length; i++)
            {
                Write("{0} ", output[i]);
            }

            //-----------------Testing-----------------------//

            ReadKey();
        }

        public static double[] AddingOne(double[] input)
        {
            var arr = input.ToList();
            arr.Insert(0, 1);
            return arr.ToArray();
        }
    }
}
