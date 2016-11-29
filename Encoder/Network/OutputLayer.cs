using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace Encoder.Network
{
    public class OutputLayer : HiddenLayer
    {
        public OutputLayer(
            int inputsCount,
            int outputsCount,
            ActivationFunction activationFunction,
            double initialWeightsRange
            ) : base(inputsCount, outputsCount, activationFunction, initialWeightsRange)
        {
        }

        public int Compute(Vector<double> inputs)
        {
            var output = Feedforward(inputs);

            return output.MaximumIndex();
        }
    }
}