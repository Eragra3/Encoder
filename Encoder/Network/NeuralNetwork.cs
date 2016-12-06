using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Security.AccessControl;
using System.Text;
using System.Threading.Tasks;
using Encoder.Mnist;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Newtonsoft.Json;

namespace Encoder.Network
{
    public class NeuralNetwork
    {
        [JsonProperty]
        protected readonly InputLayer InputLayer;
        [JsonProperty]
        protected readonly HiddenLayer[] HiddenLayers;
        [JsonProperty]
        protected readonly OutputLayer OutputLayer;

        //all layers except input
        private HiddenLayer[] _layers;
        protected HiddenLayer[] Layers
        {
            get
            {
                if (_layers != null) return _layers;

                _layers = new HiddenLayer[Sizes.Length - 1];
                for (var i = 0; i < HiddenLayers.Length; i++)
                {
                    _layers[i] = HiddenLayers[i];
                }
                _layers[_layers.Length - 1] = OutputLayer;

                return _layers;
            }
        }

        [JsonProperty]
        protected readonly int[] Sizes;

        [JsonConstructor]
        protected NeuralNetwork()
        {
        }

        public NeuralNetwork(
            ActivationFunction activationFunction,
            double initialWeightsRange,
            params int[] sizes
            )
        {
            Sizes = sizes;
            InputLayer = new InputLayer();

            HiddenLayers = new HiddenLayer[sizes.Length - 2];

            for (var i = 0; i < sizes.Length - 2; i++)
            {
                //var isLast = i == sizes.Length - 1;
                HiddenLayers[i] = new HiddenLayer(
                    sizes[i],
                    sizes[i + 1],
                    activationFunction,
                    initialWeightsRange);
            }

            var length = sizes.Length;
            OutputLayer = new OutputLayer(
                sizes[length - 2],
                sizes[length - 1],
                activationFunction,
                initialWeightsRange);
        }

        public Vector<double> FeedForward(Vector<double> input)
        {
            input = InputLayer.Feedforward(input);

            foreach (var layer in HiddenLayers)
            {
                input = layer.Feedforward(input);
            }

            var result = OutputLayer.Feedforward(input);

            return result;
        }

        public int Compute(Vector<double> input)
        {
            input = InputLayer.Feedforward(input);

            foreach (var layer in HiddenLayers)
            {
                input = layer.Feedforward(input);
            }

            var decision = OutputLayer.Compute(input);

            return decision;
        }

        public TrainingResult Train(TrainingModel trainingModel)
        {
            var trainingSet = trainingModel.TrainingSet;
            var testSet = trainingModel.TestSet;
            var validationSet = trainingModel.ValidationSet;
            var errorTreshold = trainingModel.ErrorThreshold;
            var maxEpochs = trainingModel.MaxEpochs;
            var batchSize = Math.Min(trainingModel.BatchSize, trainingModel.TrainingSet.Length);
            var learningRate = trainingModel.LearningRate;
            var momentum = trainingModel.Momentum;
            var isEncoder = trainingModel.IsEncoder;

            var isVerbose = trainingModel.IsVerbose;
            var evaluateOnEachEpoch = trainingModel.EvaluateOnEachEpoch;
            //Debugger.Launch();
            IList<double> epochErrors = new List<double>(maxEpochs);
            var epochEvaluations = new List<EvaluationModel>(maxEpochs);

            var errorSum = double.PositiveInfinity;
            var epoch = 0;

            var layersCount = Sizes.Length - 1;
            #region create nablas arrays
            var nablaWeights = new Matrix<double>[layersCount];
            var nablaBiases = new Vector<double>[layersCount];
            for (var i = 0; i < layersCount; i++)
            {
                var nextLayer = Layers[i];
                nablaBiases[i] = nextLayer.GetNewBiasesVector(true);
                nablaWeights[i] = nextLayer.GetNewWeightsMatrix(true);
            }
            #endregion

            if (isVerbose)
            {
                var activationFunctions = Layers.Select(l => l.CurrentActivationFunction.ToString()).ToArray();
                var distributions = Layers.Select(l => l.InitialWeightsRange.ToString("#0.00")).ToArray();
                Console.WriteLine("Starting with params:");
                Console.WriteLine($"\tsizes- {JsonConvert.SerializeObject(Sizes)}");
                Console.WriteLine($"\tlearning rate - {learningRate}");
                Console.WriteLine($"\tmomentum- {momentum}");
                Console.WriteLine($"\terror threshold - {errorTreshold}");
                Console.WriteLine($"\tmax epochs - {maxEpochs}");
                Console.WriteLine($"\tactivation functions - {JsonConvert.SerializeObject(activationFunctions, Formatting.None)}");
                Console.WriteLine($"\tinitial weights ranges- {JsonConvert.SerializeObject(distributions, Formatting.None)}");
            }

            InputModel sample = null;
            if (isEncoder)
            {
                Directory.CreateDirectory("encoder_logs");
                sample = trainingSet[DateTime.Now.Ticks % trainingSet.Length];
                var recreatedImage = MnistViewer.ToImage(sample.Values, 7);
                recreatedImage.Save($"encoder_logs/_original.png", ImageFormat.Png);
            }

            var prevWeightsChange = new Matrix<double>[layersCount];
            var prevBiasChange = new Vector<double>[layersCount];

            if (isVerbose && !isEncoder)
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

                var batch = HelperFunctions.RandomPermutation(trainingSet).Take(batchSize).ToList();
                //Vector<double> activationsSum = new DenseVector(_hiddenLayers[0].NeuronsCount);
                //foreach (var item in batch)
                //{
                //    var activation = item.Values;
                //    foreach (var layer in _hiddenLayers)
                //    {
                //        activation = layer.Feedforward(activation);
                //    }
                //    activationsSum += activation;
                //}
                //var avgActivations = activationsSum.Divide(batch.Count);
                foreach (var item in batch)
                {

                    //const double beta = 0.5;
                    //const double rho = 0.05;
                    //var divergence = beta * (-rho / avgActivations + (1 - rho) / (1 - avgActivations));

                    //var bpResult = Backpropagate(item.Values, item.ExpectedSolution, divergence);
                    var bpResult = Backpropagate(item.Values, item.ExpectedSolution);

                    for (var i = 0; i < nablaWeights.Length; i++)
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
                    var weightsChange = learningRate / batchSize * nablaWeights[i];
                    if (prevWeightsChange[i] != null) weightsChange += momentum * prevWeightsChange[i];
                    //L2
                    //if (true)
                    //{
                    //    weights = (1 - learningRate * 0.1) * weights;
                    //}
                    Layers[i].Weights = Layers[i].Weights - weightsChange;

                    var biasesChange = learningRate / batchSize * nablaBiases[i];
                    if (prevBiasChange[i] != null) biasesChange += momentum * prevBiasChange[i];
                    //L2
                    //if (true)
                    //{
                    //    biases = (1 - learningRate * 0.1) * biases;
                    //}
                    Layers[i].Biases = Layers[i].Biases - biasesChange;

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
                    var percentage = isEncoder
                        ? 0
                        : (epochEvaluation ?? Evaluate(testSet)).Percentage;
                    Console.WriteLine($"Epoch - {epoch}," +
                                      $" error - {errorSum.ToString("#0.000")}," +
                                      $" test - {percentage.ToString("#0.00")}");

                    if (isEncoder)
                    {
                        var recreatedData = FeedForward(sample.Values);

                        var recreatedImage = MnistViewer.ToImage(recreatedData, 7);
                        recreatedImage.Save($"encoder_logs/{epoch}.png", ImageFormat.Png);
                    }
                }

                #region dump data
                var eval = isEncoder
                        ? 0
                        : (epochEvaluation ?? Evaluate(testSet)).Percentage;

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

        //public BackpropagationResult Backpropagate(Vector<double> inputs, Vector<double> expectedOutput, Vector<double> divergences)
        public BackpropagationResult Backpropagate(Vector<double> inputs, Vector<double> expectedOutput)
        {
            var layersCount = Sizes.Length - 1;
            var nablaWeights = new Matrix<double>[layersCount];
            var nablaBiases = new Vector<double>[layersCount];
            //Debugger.Launch();

            var nets = new Vector<double>[layersCount + 1];
            var activations = new Vector<double>[layersCount + 1];
            nets[0] = inputs;
            activations[0] = inputs;
            #region get nets and activations
            var prevActivation = inputs;
            for (var i = 0; i < HiddenLayers.Length; i++)
            {
                var net = HiddenLayers[i].GetNet(prevActivation);
                nets[i + 1] = net;
                var activation = HiddenLayers[i].GetActivation(net);
                activations[i + 1] = activation;
                prevActivation = activation;
            }
            var outputNet = OutputLayer.GetNet(prevActivation);
            nets[nets.Length - 1] = outputNet;
            var outputActivation = OutputLayer.GetActivation(outputNet);
            activations[activations.Length - 1] = outputActivation;

            #endregion

            #region get output layer nablas
            var outputLayerNetDerivative = OutputLayer.ActivationFunctionPrime(outputNet);

            var delta = (outputActivation - expectedOutput).PointwiseMultiply(outputLayerNetDerivative);

            nablaBiases[nablaBiases.Length - 1] = delta;
            nablaWeights[nablaWeights.Length - 1] = delta.OuterProduct(nets[activations.Length - 2]);
            #endregion

            // skipping 0 from end
            // its reverse for loop!
            for (var layerRevIndex = 2; layerRevIndex <= _layers.Length; layerRevIndex++)
            {
                var net = nets[nets.Length - layerRevIndex];
                var prevNet = nets[nets.Length - layerRevIndex - 1];
                var layer = _layers[_layers.Length - layerRevIndex];
                var nextLayerWeights = _layers[_layers.Length - layerRevIndex + 1].Weights;
                var activationPrime = layer.ActivationFunctionPrime(net);
                //var activation = activations[activations.Length - layerRevIndex - 1];

                //the naming
                var firstPart = nextLayerWeights.Transpose().Multiply(delta);

                //if (true)
                //{
                //    firstPart += divergences;
                //}

                delta = firstPart.PointwiseMultiply(activationPrime);
                nablaBiases[nablaBiases.Length - layerRevIndex] = delta;
                nablaWeights[nablaBiases.Length - layerRevIndex] = delta.OuterProduct(prevNet);
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

            for (var i = 0; i < testData.Length; i++)
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

        public Vector<double>[][] GetFeatures()
        {
            var features = new Vector<double>[1][];

            features[0] = HiddenLayers[0].GetFeatures();
            //features[1] = _outputLayer.GetFeatures();

            return features;
        }

        public string ToJson()
        {
            return JsonConvert.SerializeObject(this, Formatting.Indented);
        }
    }
}