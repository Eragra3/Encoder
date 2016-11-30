using System;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using Encoder;
using Encoder.Experiment;
using Encoder.Mnist;
using Encoder.Network;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using TTRider.FluidCommandLine;

namespace CLI
{
    static class Program
    {
        private const string DATA_PATH = "../";
        private const string TEST_DATA_PATH = DATA_PATH + "TestData";
        private const string TRAINING_DATA_PATH = DATA_PATH + "TestData";
        private const string VALIDATION_PATH = DATA_PATH + "ValidationData";

        static void Main(string[] args)
        {
            if (!Optimizer.TryUseNativeMKL())
            {
                Console.Error.WriteLine("Could not use MKL native library");
            }

            var command = Command.Help;
            var nnJsonPath = "";
            var isVerbose = false;
            var outputPath = "";
            var imagePath = "";
            var print = false;
            var evaluate = false;
            var dump = false;
            var isEncoder = false;

            var normalize = false;

            //mlp params
            int[] layersSizes = { 70, 200, 10 };
            var learningRate = 0.5;
            var momentum = 0.9;
            double errorThreshold = 0;
            var batchSize = 20;
            var activationFunction = ActivationFunction.Sigmoid;
            var normalStDev = 0.5;

            var imageWidth = 10;

            var maxEpochs = 200;

            string experimentValues = null;
            var experiment = Experiment.LearningRate;
            var repetitions = 3;

            ICommandLine commandLine = CommandLine
                .Help("h")
                .Help("help")
                .Command("test", () => command = Command.Test, true, "Test your MLP")
                    .DefaultParameter("mlp", json => nnJsonPath = json, "MLP data in json format", "Json")
                    .Parameter("image", path => imagePath = path, "Path to image", "Path to image")
                    .Option("v", () => isVerbose = true, "Explain what is happening")
                    .Option("verbose", () => isVerbose = true, "Explain what is happening")
                    .Option("e", () => evaluate = true, "Evaluate using MNIST dataset")
                    .Option("evaluate", () => evaluate = true, "Evaluate using MNIST dataset")
                    .Option("n", () => normalize = true, "Normalize input")
                    .Option("normalize", () => normalize = true, "Normalize input")
                    .Option("e", () => isEncoder = true, "Use encoder mode")
                    .Option("encoder", () => isEncoder = true, "Use encoder mode")
                .Command("train", () => command = Command.Train, "Train new MLP")
                    .DefaultParameter("output", path => outputPath = path, "Output file to save trained mlp")
                    .Parameter("sizes", sizes => layersSizes = JsonConvert.DeserializeObject<int[]>(sizes), "Number of layer and its sizes, default to [70,5,10]", "Sizes")
                    .Parameter("learning-rate", val => learningRate = double.Parse(val, CultureInfo.InvariantCulture), "Learning rate")
                    .Parameter("momentum", val => momentum = double.Parse(val, CultureInfo.InvariantCulture), "Momenum parameter")
                    .Parameter("error-threshold", val => errorThreshold = double.Parse(val, CultureInfo.InvariantCulture), "Error threshold to set learning stop criteria")
                    .Parameter("max-epochs", val => maxEpochs = int.Parse(val), "Program will terminate learning if reaches this epoch")
                    .Parameter("batch-size", val => batchSize = int.Parse(val), "Batch size")
                    .Parameter("activation", val => activationFunction = ParseActivationFunction(val), "Activation function, (sigmoid, tanh)")
                    .Parameter("normal", val => normalStDev = double.Parse(val, CultureInfo.InvariantCulture), "Initial weights normal distribution standard deviation")
                    .Option("v", () => isVerbose = true, "Explain what is happening")
                    .Option("verbose", () => isVerbose = true, "Explain what is happening")
                    .Option("d", () => dump = true, "Dump training data")
                    .Option("dump", () => dump = true, "Dump training data")
                    .Option("n", () => normalize = true, "Normalize input")
                    .Option("normalize", () => normalize = true, "Normalize input")
                    .Option("e", () => isEncoder = true, "Use encoder mode")
                    .Option("encoder", () => isEncoder = true, "Use encoder mode")
                .Command("view", () => command = Command.View, "Show MNIST image")
                    .DefaultParameter("path", path => imagePath = path, "Path to image")
                    .Option("p", () => print = true, "Display grayscale interpretation")
                    .Option("print", () => print = true, "Display grayscale interpretation")
                    .Option("n", () => normalize = true, "Normalize input")
                    .Option("normalize", () => normalize = true, "Normalize input")
                .Command("experiment", () => command = Command.Experiment, "Run experiment")
                    .DefaultParameter("output", path => outputPath = path, "Path to save data")
                    .Parameter("values", val => experimentValues = val, "Values to test in experiment", "Experiment values")
                    .Parameter("experiment", val => experiment = ParseExperimentType(val), "Momenum parameter")
                    .Parameter("sizes", sizes => layersSizes = JsonConvert.DeserializeObject<int[]>(sizes), "Number of layer and its sizes, default to [70,5,10]", "Sizes")
                    .Parameter("learning-rate", val => learningRate = double.Parse(val, CultureInfo.InvariantCulture), "Learning rate")
                    .Parameter("momentum", val => momentum = double.Parse(val, CultureInfo.InvariantCulture), "Momenum parameter")
                    .Parameter("error-threshold", val => errorThreshold = double.Parse(val, CultureInfo.InvariantCulture), "Error threshold to set learning stop criteria")
                    .Parameter("max-epochs", val => maxEpochs = int.Parse(val), "Program will terminate learning if reaches this epoch")
                    .Parameter("batch-size", val => batchSize = int.Parse(val), "Batch size")
                    .Parameter("activation", val => activationFunction = ParseActivationFunction(val), "Activation function, (sigmoid, tanh)")
                    .Parameter("normal", val => normalStDev = double.Parse(val, CultureInfo.InvariantCulture), "Initial weights normal distribution standard deviation")
                    .Parameter("repetitions", val => repetitions = int.Parse(val, CultureInfo.InvariantCulture), "Number of repetitions for each value in experiment")
                    .Option("v", () => isVerbose = true, "Explain what is happening")
                    .Option("verbose", () => isVerbose = true, "Explain what is happening")
                    .Option("n", () => normalize = true, "Normalize input")
                    .Option("normalize", () => normalize = true, "Normalize input")
                    .Option("e", () => isEncoder = true, "Use encoder mode")
                    .Option("encoder", () => isEncoder = true, "Use encoder mode")
                .Command("features", () => command = Command.Features, "Print features")
                    .Parameter("mlp", json => nnJsonPath = json, "MLP data in json format", "Json")
                    .Parameter("output", path => outputPath = path, "Path to save features")
                    .Parameter("width", val => imageWidth = int.Parse(val), "Input width to display feature as image")
                .End();

            commandLine.Run(args);

            switch (command)
            {
                case Command.Train:
                    {
                        try
                        {
                            File.Create(outputPath).Close();
                        }
                        catch (Exception)
                        {
                            Console.WriteLine($"Path is invalid");
                            return;
                        }

                        var options = new NeuralNetworkOptions(
                            learningRate,
                            momentum,
                            errorThreshold,
                            layersSizes,
                            TRAINING_DATA_PATH,
                            VALIDATION_PATH,
                            TEST_DATA_PATH,
                            maxEpochs,
                            isVerbose,
                            batchSize,
                            activationFunction,
                            normalStDev,
                            dump,
                            normalize,
                            isEncoder
                            );

                        var trainingResult = MnistTrainer.TrainOnMnist(options);

                        var mlp = trainingResult.NeuralNetwork;

                        File.WriteAllText(outputPath, mlp.ToJson());

                        if (dump)
                        {
                            var fi = new FileInfo(outputPath);
                            var directory = fi.Directory?.FullName ?? "";
                            var fileName = Path.GetFileNameWithoutExtension(outputPath);
                            directory += $"/{fileName}_";
                            ExperimentVisualization.GenerateErrorPlot(trainingResult, $"{directory}error");

                            if (!isEncoder)
                            {
                                ExperimentVisualization.GenerateEvaluationPlot(trainingResult, $"{directory}evaluation");
                            }
                        }

                        break;
                    }
                case Command.Test:
                    {
                        NeuralNetwork mlp;
                        try
                        {
                            var json = File.ReadAllText(nnJsonPath);
                            mlp = JsonConvert.DeserializeObject<NeuralNetwork>(json);
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine(e);
                            Console.WriteLine($"Path is invalid");
                            return;
                        }

                        if (!string.IsNullOrEmpty(imagePath))
                        {
                            if (!File.Exists(imagePath))
                            {
                                Console.WriteLine($"File {imagePath} does not exist!");
                                return;
                            }

                            var image = MnistParser.ReadImage(imagePath, normalize, isEncoder);

                            var decision = mlp.Compute(image.Values);

                            Console.WriteLine($"Result - {decision}");
                            Console.WriteLine($"Expected - {image.Label}");
                        }

                        if (evaluate)
                        {
                            var testData = MnistParser.ReadAll(TEST_DATA_PATH, normalize, isEncoder);

                            var evaluation = mlp.Evaluate(testData);

                            Console.WriteLine($"Solutions - {evaluation.Correct} / {evaluation.All}");
                            Console.WriteLine($"Fitness - {evaluation.Percentage}");
                        }

                        break;
                    }
                case Command.View:
                    {
                        if (string.IsNullOrEmpty(imagePath))
                        {
                            Console.WriteLine($"Path to image not set");
                            return;
                        }
                        if (!File.Exists(imagePath))
                        {
                            Console.WriteLine($"File {imagePath} does not exist!");
                            return;
                        }

                        var model = MnistParser.ReadImage(imagePath, normalize, isEncoder);

                        var modelDump = MnistViewer.Dump(model);
                        Console.WriteLine(modelDump);

                        if (print)
                        {
                            var modelMatrix = MnistViewer.ToMatrix(model.Values, model.Width);
                            Console.Write(modelMatrix);
                        }
                        break;
                    }
                case Command.Help:
                    commandLine.Run("help");
                    break;
                case Command.Experiment:
                    {
                        var options = new NeuralNetworkOptions(
                            learningRate,
                            momentum,
                            errorThreshold,
                            layersSizes,
                            TRAINING_DATA_PATH,
                            VALIDATION_PATH,
                            TEST_DATA_PATH,
                            maxEpochs,
                            isVerbose,
                            batchSize,
                            activationFunction,
                            normalStDev,
                            true,
                            normalize,
                            isEncoder
                            );

                        switch (experiment)
                        {
                            case Experiment.LearningRate:
                                {
                                    var values = JsonConvert.DeserializeObject<double[]>(experimentValues);
                                    ExperimentRunner.RunLearningRateExperiment(
                                        values,
                                        options,
                                        repetitions,
                                        outputPath
                                        );
                                    break;
                                }
                            case Experiment.ActivationFunction:
                                {
                                    var values = JsonConvert.DeserializeObject<ActivationFunction[]>(experimentValues);
                                    ExperimentRunner.RunActivationFunctionExperiment(
                                        values,
                                        options,
                                        repetitions,
                                        outputPath
                                        );
                                    break;
                                }
                            case Experiment.Momentum:
                                {
                                    var values = JsonConvert.DeserializeObject<double[]>(experimentValues);
                                    ExperimentRunner.RunMomentumExperiment(
                                        values,
                                        options,
                                        repetitions,
                                        outputPath
                                        );
                                    break;
                                }
                            case Experiment.NormalDistStDev:
                                {
                                    var values = JsonConvert.DeserializeObject<double[]>(experimentValues);
                                    ExperimentRunner.RunInitialWeightsRangeExperiment(
                                        values,
                                        options,
                                        repetitions,
                                        outputPath
                                        );
                                    break;
                                }
                            default:
                                throw new ArgumentOutOfRangeException();
                        }
                        break;
                    }
                case Command.Features:
                    {
                        NeuralNetwork mlp;
                        try
                        {
                            var json = File.ReadAllText(nnJsonPath);
                            mlp = JsonConvert.DeserializeObject<NeuralNetwork>(json);
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine(e);
                            Console.WriteLine($"Path is invalid");
                            return;
                        }

                        var layerFeatures = mlp.GetFeatures();

                        Directory.Delete(outputPath, true);
                        Directory.CreateDirectory(outputPath);
                        for (int layerIndex = 0; layerIndex < layerFeatures.Length; layerIndex++)
                        {
                            var features = layerFeatures[layerIndex];
                            var path = $"{outputPath}/{layerIndex}";
                            Directory.CreateDirectory(path);

                            for (int i = 0; i < features.Length; i++)
                            {
                                var feature = features[i];
                                var image = MnistViewer.ToImage(feature, imageWidth);

                                image.Save($"{path}/{i}.png", ImageFormat.Png);
                            }
                        }

                        break;
                    }
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        private static ActivationFunction ParseActivationFunction(string val)
        {
            val = val.ToLower();
            switch (val)
            {
                case "sigmoid":
                    return ActivationFunction.Sigmoid;
                case "tanh":
                    return ActivationFunction.Tanh;
                default:
                    throw new ArgumentException();
            }
        }

        private static Experiment ParseExperimentType(string val)
        {
            val = val.ToLower();
            switch (val)
            {
                case "learningrate":
                    return Experiment.LearningRate;
                case "activationfunction":
                    return Experiment.ActivationFunction;
                case "momentum":
                    return Experiment.Momentum;
                case "standarddeviation":
                    return Experiment.NormalDistStDev;
                default:
                    throw new ArgumentException();
            }
        }
    }
}
