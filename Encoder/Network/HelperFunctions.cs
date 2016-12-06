using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace Encoder.Network
{
    public static class HelperFunctions
    {
        private static readonly Random Random = new Random();

        public static Vector<double> Sigmoid(Vector<double> input)
        {
            var result = input.Map(x => 1.0 / (1.0 + Math.Exp(-x)), Zeros.Include);
            return result;
        }

        public static Vector<double> SigmoidPrime(Vector<double> input)
        {
            var sigmoid = Sigmoid(input);

            var result = sigmoid.PointwiseMultiply(1 - sigmoid);

            return result;
        }

        public static Vector<double> Tanh(Vector<double> input)
        {
            var result = input.Map(Math.Tanh);
            return result;
        }

        public static Vector<double> TanhPrime(Vector<double> input)
        {
            var result = input.Map(x => 1 - Math.Pow(Math.Tanh(x), 2));

            return result;
        }

        public static T[] RandomPermutation<T>(IEnumerable<T> sequence)
        {
            var retArray = sequence.ToArray();


            for (var i = 0; i < retArray.Length - 1; i += 1)
            {
                var swapIndex = Random.Next(i, retArray.Length);
                if (swapIndex == i) continue;
                var temp = retArray[i];
                retArray[i] = retArray[swapIndex];
                retArray[swapIndex] = temp;
            }

            return retArray;
        }
    }
}
