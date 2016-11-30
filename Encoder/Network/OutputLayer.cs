using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json;

namespace Encoder.Network
{
    public class OutputLayer : HiddenLayer
    {
        [JsonConstructor]
        private OutputLayer()
        {

        }

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


        //public new Vector<double>[] GetFeatures()
        //{
        //    var allFeatures = new Vector<double>[_neuronsCount];

        //    for (var i = 0; i < _neuronsCount; i++)
        //    {
        //        var solution = new SparseVector(_neuronsCount) { [i] = 1 };

        //        var activation = solution * Weights;

        //        var norm = activation.Row(i).L2Norm();
        //        var features = activation.Row(i).Divide(norm);

        //        var max = features.Maximum();
        //        var min = features.Minimum();
        //        features.MapInplace(v => (v - min) / (max - min));

        //        allFeatures[i] = features;
        //    }

        //    return allFeatures;
        //}
    }
}