using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;

namespace Encoder.Network
{
    public class Encoder : NeuralNetwork
    {
        [JsonConstructor]
        protected Encoder()
        {

        }

        public Encoder(
            ActivationFunction activationFunction,
            double initialWeightsRange,
            params int[] sizes
            ) : base(activationFunction, initialWeightsRange, sizes)
        {
        }

        public new Vector<double> Compute(Vector<double> input)
        {
            input = InputLayer.Feedforward(input);

            foreach (var layer in HiddenLayers)
            {
                input = layer.Feedforward(input);
            }

            var decision = OutputLayer.Feedforward(input);

            return decision;
        }
    }
}
