using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;

namespace Encoder.Network
{
    public class NeuralNetwork
    {
        [JsonProperty]
        private readonly InputLayer _inputLayer;
        [JsonProperty]
        private readonly HiddenLayer[] _hiddenLayers;
        [JsonProperty]
        private readonly OutputLayer _outputLayer;

        //all layers except input
        private readonly HiddenLayer[] _layers;

        [JsonProperty]
        private readonly int[] _sizes;

        [JsonConstructor]
        private NeuralNetwork()
        {
        }

        public NeuralNetwork(
            ActivationFunction activationFunction,
            double initialWeightsRange,
            params int[] sizes
            )
        {
            _sizes = sizes;
            _inputLayer = new InputLayer();

            _hiddenLayers = new HiddenLayer[sizes.Length - 2];

            for (var i = 0; i < sizes.Length - 2; i++)
            {
                //var isLast = i == sizes.Length - 1;
                _hiddenLayers[i] = new HiddenLayer(
                    sizes[i],
                    sizes[i + 1],
                    activationFunction,
                    initialWeightsRange);
            }

            var length = sizes.Length;
            _outputLayer = new OutputLayer(
                sizes[length - 2],
                sizes[length - 1],
                activationFunction,
                initialWeightsRange);

            _layers = new HiddenLayer[_sizes.Length - 1];
            for (int i = 0; i < _hiddenLayers.Length; i++)
            {
                _layers[i] = _hiddenLayers[i];
            }
            _layers[_layers.Length - 1] = _outputLayer;
        }

        public int Compute(Vector<double> input)
        {
            input = _inputLayer.Feedforward(input);

            foreach (var layer in _hiddenLayers)
            {
                input = layer.Feedforward(input);
            }

            var decision = _outputLayer.Compute(input);

            return decision;
        }

        public TrainingResult Train(TrainingModel trainingModel)
        {
            var trainingSet = trainingModel.TrainingSet;
            var testSet = trainingModel.TestSet;
            var validationSet = trainingModel.ValidationSet;
            var errorTreshold = trainingModel.ErrorThreshold;
            var maxEpochs = trainingModel.MaxEpochs;
            var batchSize = trainingModel.BatchSize;
            var learningRate = trainingModel.LearningRate;
            var momentum = trainingModel.Momentum;

            var isVerbose = trainingModel.IsVerbose;
            var evaluateOnEachEpoch = trainingModel.EvaluateOnEachEpoch;

            IList<double> epochErrors = new List<double> { 0 };
            var epochEvaluations = new List<EvaluationModel>();

            var errorSum = double.PositiveInfinity;
            var epoch = 0;

            var layersCount = _sizes.Length - 1;
            #region create nablas arrays
            var nablaWeights = new Matrix<double>[layersCount];
            var nablaBiases = new Vector<double>[layersCount];
            for (var i = 0; i < layersCount; i++)
            {
                var nextLayer = _layers[i];
                nablaBiases[i] = nextLayer.GetNewBiasesVector(true);
                nablaWeights[i] = nextLayer.GetNewWeightsMatrix(true);
            }
            #endregion

            if (isVerbose)
            {
                var activationFunctions = _layers.Select(l => l.CurrentActivationFunction.ToString()).ToArray();
                var distributions = _layers.Select(l => l.InitialWeightsRange.ToString("#0.00")).ToArray();
                Console.WriteLine("Starting with params:");
                Console.WriteLine($"\tsizes- {JsonConvert.SerializeObject(_sizes)}");
                Console.WriteLine($"\tlearning rate - {learningRate}");
                Console.WriteLine($"\tmomentum- {momentum}");
                Console.WriteLine($"\terror threshold - {errorTreshold}");
                Console.WriteLine($"\tmax epochs - {maxEpochs}");
                Console.WriteLine($"\tactivation functions - {JsonConvert.SerializeObject(activationFunctions, Formatting.None)}");
                Console.WriteLine($"\tinitial weights ranges- {JsonConvert.SerializeObject(distributions, Formatting.None)}");
            }

            var prevWeightsChange = new Matrix<double>[layersCount];
            var prevBiasChange = new Vector<double>[layersCount];

            if (isVerbose)
            {
                var initialPercentage = Evaluate(testSet).Percentage;
                Console.WriteLine($"Initial state, {initialPercentage.ToString("#0.00")}");
            }

            #region log data
            //log data
            var path = "log.csv";

            var log = new StringBuilder("sep=|");
            log.AppendLine();
            log.Append("epoch|evaluation_0|error_0");
            log.AppendLine();
            #endregion

            while (errorSum > errorTreshold && epoch < maxEpochs)
            {
                epoch++;
                errorSum = 0;

                foreach (var item in HelperFunctions.RandomPermutation(trainingSet).Take(batchSize))
                {
                    var bpResult = Backpropagate(item.Values, item.ExpectedSolution);

                    for (var i = 0; i < layersCount - 1; i++)
                    {
                        nablaBiases[i] = bpResult.Biases[i] + nablaBiases[i];
                        nablaWeights[i] = bpResult.Weights[i] + nablaWeights[i];
                    }

                    var solution = bpResult.Solution;
                    var expectedSolution = item.ExpectedSolution;

                    errorSum += solution.Map2((y, o) => Math.Abs(y - o), expectedSolution).Sum();
                }
                errorSum /= batchSize;

                #region update parameters
                for (var i = 0; i < layersCount; i++)
                {
                    var weights = _layers[i].Weights;
                    var weightsChange = learningRate / batchSize * nablaWeights[i];
                    if (prevWeightsChange[i] != null) weightsChange += momentum * prevWeightsChange[i];
                    _layers[i].Weights = weights - weightsChange;

                    var biases = _layers[i].Biases;
                    var biasesChange = learningRate / batchSize * nablaBiases[i];
                    if (prevBiasChange[i] != null) biasesChange += momentum * prevBiasChange[i];
                    _layers[i].Biases = biases - biasesChange;

                    prevWeightsChange[i] = weightsChange;
                    prevBiasChange[i] = biasesChange;
                }
                #endregion

                EvaluationModel epochEvaluation = null;
                if (evaluateOnEachEpoch)
                {
                    epochEvaluation = Evaluate(testSet);
                    epochEvaluations.Add(epochEvaluation);
                }

                if (isVerbose)
                {
                    var percentage = (epochEvaluation ?? Evaluate(testSet)).Percentage;
                    Console.WriteLine($"Epoch - {epoch}," +
                                      $" error - {errorSum.ToString("#0.000")}," +
                                      $" test - {percentage.ToString("#0.00")}");
                }

                #region dump data
                var eval = (epochEvaluation ?? Evaluate(testSet)).Percentage;

                log.AppendLine(epoch + "|" + eval + "|" + errorSum);
                #endregion

                #region set nablas to zeroes
                for (var i = 0; i < layersCount; i++)
                {
                    nablaBiases[i].Clear();
                    nablaWeights[i].Clear();
                }
                #endregion

                epochErrors.Add(errorSum);
            }

            #region log data
            File.WriteAllText(path, log.ToString());
            #endregion

            var trainingResult = new TrainingResult
            {
                NeuralNetwork = this,
                Epochs = epoch,
                EpochErrors = epochErrors.ToArray(),
                Evaluations = epochEvaluations.ToArray()
            };

            return trainingResult;
        }

        public BackpropagationResult Backpropagate(Vector<double> inputs, Vector<double> expectedOutput)
        {
            var layersCount = _sizes.Length - 1;
            var nablaWeights = new Matrix<double>[layersCount];
            var nablaBiases = new Vector<double>[layersCount];

            var nets = new Vector<double>[layersCount + 1];
            var activations = new Vector<double>[layersCount + 1];
            nets[0] = inputs;
            activations[0] = inputs;
            #region get nets and activations
            var prevActivation = inputs;
            for (var i = 0; i < _hiddenLayers.Length; i++)
            {
                var net = _hiddenLayers[i].GetNet(prevActivation);
                nets[i + 1] = net;
                var activation = _hiddenLayers[i].GetActivation(net);
                activations[i + 1] = activation;
                prevActivation = activation;
            }
            var outputNet = _outputLayer.GetNet(prevActivation);
            nets[layersCount] = outputNet;
            var outputActivation = _outputLayer.GetActivation(outputNet);
            activations[layersCount] = outputActivation;

            #endregion

            #region get output layer nablas
            var outputLayerNetDerivative = _outputLayer.ActivationFunctionPrime(outputNet);

            var delta = (outputActivation - expectedOutput).PointwiseMultiply(outputLayerNetDerivative);

            nablaBiases[nablaBiases.Length - 1] = delta;
            nablaWeights[nablaWeights.Length - 1] = delta.OuterProduct(outputActivation);
            #endregion

            for (var layerIndex = _hiddenLayers.Length - 1; layerIndex >= 0; layerIndex--)
            {
                var layer = _hiddenLayers[layerIndex];
                var activation = activations[layerIndex];
                var net = nets[layerIndex + 1];
                var activationPrime = layer.ActivationFunctionPrime(net);
                var nextLayerWeights = layerIndex + 1 == _hiddenLayers.Length
                    ? _outputLayer.Weights
                    : _hiddenLayers[layerIndex + 1].Weights;

                delta = nextLayerWeights.Transpose().Multiply(delta).PointwiseMultiply(activationPrime);
                nablaBiases[layerIndex] = delta;
                nablaWeights[layerIndex] = delta.OuterProduct(activation);
            }

            var result = new BackpropagationResult
            {
                Biases = nablaBiases,
                Weights = nablaWeights,
                Input = inputs,
                Solution = activations.Last()
            };

            return result;
        }

        public EvaluationModel Evaluate(InputModel[] testData)
        {
            var correctSolutions = 0;

            for (int i = 0; i < testData.Length; i++)
            {
                var model = testData[i];

                var decision = Compute(model.Values);

                if (decision == model.Label) correctSolutions++;
            }

            var result = new EvaluationModel
            {
                Correct = correctSolutions,
                All = testData.Length
            };

            return result;
        }

        public string ToJson()
        {
            return JsonConvert.SerializeObject(this, Formatting.Indented);
        }

        public static NeuralNetwork FromJson(string json)
        {
            return JsonConvert.DeserializeObject<NeuralNetwork>(json);
        }
    }
}