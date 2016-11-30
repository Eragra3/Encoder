using System;
using System.Globalization;
using System.IO;
using Encoder;
using Encoder.Experiment;
using Encoder.Mnist;
using Encoder.Network;
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

            Command command = Command.Help;
            string nnJsonPath = "";
            bool isVerbose = false;
            string outputPath = "";
            string imagePath = "";
            bool print = false;
            bool evaluate = false;

            bool normalize = false;

            //mlp params
            int[] layersSizes = { 70, 200, 10 };
            double learningRate = 7;
            double momentum = 0.00;
            double errorThreshold = 0;
            int batchSize = 20;
            ActivationFunction activationFunction = ActivationFunction.Sigmoid;
            double normalStDev = 0.5;

            int maxEpochs = 200;

            string experimentValues = null;
            Experiment experiment = Experiment.LearningRate;
            int repetitions = 3;

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
                    .Option("n", () => normalize = true, "Normalize input")
                    .Option("normalize", () => normalize = true, "Normalize input")
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
                            false,
                            normalize
                            );

                        var trainingResult = MnistTrainer.TrainOnMnist(options);

                        var mlp = trainingResult.NeuralNetwork;

                        File.WriteAllText(outputPath, mlp.ToJson());

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

                            var image = MnistParser.ReadImage(imagePath, normalize);

                            var decision = mlp.Compute(image.Values);

                            Console.WriteLine($"Result - {decision}");
                            Console.WriteLine($"Expected - {image.Label}");
                        }

                        if (evaluate)
                        {
                            var testData = MnistParser.ReadAll(TEST_DATA_PATH, normalize);

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

                        var model = MnistParser.ReadImage(imagePath, normalize);

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
                            normalize
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
                    }
                    break;
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
