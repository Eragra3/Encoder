
using System;
using System.Diagnostics;
using Encoder.Serialization;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json;
using Matrix = MathNet.Numerics.LinearAlgebra.Double.Matrix;
using Vector = MathNet.Numerics.LinearAlgebra.Double.Vector;

namespace Encoder.Network
{
    public class HiddenLayer : Layer
    {
        /// <summary>
        /// Matrix
        /// dimensions: O x I
        /// </summary>
        /// <remarks>
        /// O == N
        /// Output count is same as neuron count
        /// </remarks>
        [JsonProperty]
        [JsonConverter(typeof(MatrixConverter))]
        public Matrix<double> Weights;

        /// <summary>
        /// Vector
        /// dimensions: O x 1
        /// </summary>
        [JsonProperty]
        [JsonConverter(typeof(VectorConverter))]
        public Vector<double> Biases;

        [JsonProperty]
        protected readonly int _inputsCount;
        [JsonProperty]
        protected readonly int _neuronsCount;

        public ActivationFunction CurrentActivationFunction { get; }

        public readonly double InitialWeightsRange;

        protected IContinuousDistribution _distribution;
        protected IContinuousDistribution CurrentDistribution
        {
            get
            {
                if (_distribution != null) return _distribution;

                _distribution = new ContinuousUniform(-InitialWeightsRange, InitialWeightsRange);

                return _distribution;
            }
            set { _distribution = value; }
        }

        [JsonConstructor]
        protected HiddenLayer()
        {

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputsCount">Number of inputs to each vector</param>
        /// <param name="outputsCount">Same as neuron count in this layer</param>
        /// <param name="activationFunction"></param>
        /// <param name="initialWeightsRange"></param>
        public HiddenLayer(
            int inputsCount,
            int outputsCount,
            ActivationFunction activationFunction,
            double initialWeightsRange
            )
        {
            CurrentActivationFunction = activationFunction;
            _inputsCount = inputsCount;
            _neuronsCount = outputsCount;
            InitialWeightsRange = initialWeightsRange;

            Weights = GetNewWeightsMatrix(false);
            Biases = GetNewBiasesVector(false);
        }

        public Matrix GetNewWeightsMatrix(bool allZeroes)
        {
            if (allZeroes) return new DenseMatrix(_neuronsCount, _inputsCount);
            return DenseMatrix.CreateRandom(_neuronsCount, _inputsCount, CurrentDistribution);
        }

        public Vector GetNewBiasesVector(bool allZeroes)
        {
            if (allZeroes) return new DenseVector(_neuronsCount);
            return DenseVector.CreateRandom(_neuronsCount, CurrentDistribution);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputs">Vector dimensions: I x 1</param>
        /// <returns></returns>
        public override Vector<double> Feedforward(Vector<double> inputs)
        {
            var output = GetNet(inputs);

            return GetActivation(output);
        }

        public Vector<double> GetNet(Vector<double> inputs)
        {
            //(O x I * I x 1) o 0 x 1 = O x 1
            return Weights * inputs + Biases;
        }

        /// <summary>
        /// Takes net as input (this layer output before activation function)
        /// </summary>
        /// <param name="output"></param>
        /// <returns></returns>
        public Vector<double> GetActivation(Vector<double> output)
        {
            return ActivationFunction(output);
        }

        public Vector<double> ActivationFunction(Vector<double> output)
        {
            switch (CurrentActivationFunction)
            {
                case Network.ActivationFunction.Sigmoid:
                    return HelperFunctions.Sigmoid(output);
                case Network.ActivationFunction.Tanh:
                    return HelperFunctions.Tanh(output);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public Vector<double> ActivationFunctionPrime(Vector<double> output)
        {
            switch (CurrentActivationFunction)
            {
                case Network.ActivationFunction.Sigmoid:
                    return HelperFunctions.SigmoidPrime(output);
                case Network.ActivationFunction.Tanh:
                    return HelperFunctions.TanhPrime(output);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public Vector<double>[] GetFeatures()
        {
            var allFeatures = new Vector<double>[_neuronsCount];

            for (var i = 0; i < Weights.RowCount; i++)
            {
                var norm = Weights.Row(i).L2Norm();
                var features = Weights.Row(i).Divide(norm);

                allFeatures[i] = features;
            }

            return allFeatures;
        }
    }
}