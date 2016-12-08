using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encoder.Mnist;
using Encoder.Network;
using Newtonsoft.Json;

namespace Encoder.Experiment
{
    public static class ExperimentRunner
    {
        public static void RunLearningRateExperiment(
            double[] learningRates,
            NeuralNetworkOptions options,
            int repetitions,
            string logPath
            )
        {
            //disable early learning end
            options.ErrorThreshold = 0;
            var isVerbose = options.IsVerbose;

            var mainDir = logPath.Split('/')[0];
            if (Directory.Exists(mainDir)) ClearDirectory(mainDir);
            Directory.CreateDirectory(mainDir);

            for (var i = 0; i < learningRates.Length; i++)
            {
                var learningRate = learningRates[i];

                Console.WriteLine($"Running experiment for {learningRate}");

                var trainingOptions = new NeuralNetworkOptions(
                    learningRate,
                    options.Momentum,
                    options.ErrorThreshold,
                    options.Sizes,
                    options.TrainingPath,
                    options.ValidationPath,
                    options.TestPath,
                    options.MaxEpochs,
                    options.IsVerbose,
                    options.BatchSize,
                    options.ActivationFunction,
                    options.InitialWeightsRange,
                    true,
                    options.NormalizeInput,
                    options.IsEncoder,
                    options.Lambda, options.TakeBest
                    );

                #region dump used params
                //lel
                var dumpling = JsonConvert.SerializeObject(options, Formatting.Indented);
                File.WriteAllText(logPath + ".log", dumpling);
                #endregion

                var trainingResponses = new TrainingResult[repetitions];

                var runLogPath = logPath + "/" + learningRate;
                Directory.CreateDirectory(runLogPath);

                //gather data
                for (var j = 0; j < repetitions; j++)
                {
                    var mlp = new NeuralNetwork(options.ActivationFunction, options.InitialWeightsRange, options.Sizes);

                    var trainingResponse = MnistTrainer.TrainOnMnist(trainingOptions);
                    trainingResponses[j] = trainingResponse;

                    File.WriteAllText($"{runLogPath}/{learningRate}_{j}.json", trainingResponse.NeuralNetwork.ToJson());
                }

                var fileName = logPath + "_" + learningRate;

                //log data
                var path = fileName + ".csv";

                //File.Create(path);

                var log = new StringBuilder("sep=|");
                log.AppendLine();
                log.Append("epoch");
                for (var j = 0; j < trainingResponses.Length; j++)
                {
                    log.Append("|evaluation_" + j + "|error_" + j);
                }
                log.AppendLine();
                for (var j = 0; j < trainingResponses[0].Epochs; j++)
                {
                    log.Append(j);
                    for (var n = 0; n < trainingResponses.Length; n++)
                    {
                        var result = trainingResponses[n];
                        log.Append("|" + result.Evaluations[j].Percentage + "|" + result.EpochErrors[j]);
                    }
                    log.AppendLine();
                }
                File.WriteAllText(path, log.ToString());

                #region dump plot
                if (!options.IsEncoder)
                {
                    ExperimentVisualization.GenerateEvaluationPlot(trainingResponses, fileName, fileName);
                }
                ExperimentVisualization.GenerateErrorPlot(trainingResponses, logPath + "_error_" + learningRate, fileName);
                #endregion
            }
        }

        public static void RunInitialWeightsRangeExperiment(
            double[] initialWeightsRanges,
            NeuralNetworkOptions options,
            int repetitions,
            string logPath
            )
        {
            //disable early learning end
            options.ErrorThreshold = 0;
            var isVerbose = options.IsVerbose;

            var mainDir = logPath.Split('/')[0];
            if (Directory.Exists(mainDir)) ClearDirectory(mainDir);
            Directory.CreateDirectory(mainDir);

            for (var i = 0; i < initialWeightsRanges.Length; i++)
            {
                var initialWeightsRange = initialWeightsRanges[i];

                Console.WriteLine($"Running experiment for {initialWeightsRange}");

                var trainingOptions = new NeuralNetworkOptions(
                    options.LearningRate,
                    options.Momentum,
                    options.ErrorThreshold,
                    options.Sizes,
                    options.TrainingPath,
                    options.ValidationPath,
                    options.TestPath,
                    options.MaxEpochs,
                    options.IsVerbose,
                    options.BatchSize,
                    options.ActivationFunction,
                    initialWeightsRange,
                    true,
                    options.NormalizeInput,
                    options.IsEncoder,
                    options.Lambda, options.TakeBest
                    );

                #region dump used params
                //lel
                var dumpling = JsonConvert.SerializeObject(options, Formatting.Indented);
                File.WriteAllText(logPath + ".log", dumpling);
                #endregion

                var trainingResponses = new TrainingResult[repetitions];

                var runLogPath = logPath + "/" + initialWeightsRange;
                Directory.CreateDirectory(runLogPath);

                //gather data
                for (var j = 0; j < repetitions; j++)
                {
                    var trainingResponse = MnistTrainer.TrainOnMnist(trainingOptions);
                    trainingResponses[j] = trainingResponse;

                    File.WriteAllText($"{runLogPath}/{initialWeightsRange}_{j}.json", trainingResponse.NeuralNetwork.ToJson());
                }

                var fileName = logPath + "_" + initialWeightsRange;

                //log data
                var path = fileName + ".csv";

                //File.Create(path);

                var log = new StringBuilder("sep=|");
                log.AppendLine();
                log.Append("epoch");
                for (var j = 0; j < trainingResponses.Length; j++)
                {
                    log.Append("|evaluation_" + j + "|error_" + j);
                }
                log.AppendLine();
                for (var j = 0; j < trainingResponses[0].Epochs; j++)
                {
                    log.Append(j);
                    for (var n = 0; n < trainingResponses.Length; n++)
                    {
                        var result = trainingResponses[n];
                        log.Append("|" + result.Evaluations[j].Percentage + "|" + result.EpochErrors[j]);
                    }
                    log.AppendLine();
                }
                File.WriteAllText(path, log.ToString());

                #region dump plot
                ExperimentVisualization.GenerateEvaluationPlot(trainingResponses, fileName, fileName);
                ExperimentVisualization.GenerateErrorPlot(trainingResponses, logPath + "_error_" + initialWeightsRange, fileName);
                #endregion
            }
        }

        public static void RunActivationFunctionExperiment(
            ActivationFunction[] activatonFunctions,
            NeuralNetworkOptions options,
            int repetitions,
            string logPath
            )
        {
            //disable early learning end
            options.ErrorThreshold = 0;
            var isVerbose = options.IsVerbose;

            var mainDir = logPath.Split('/')[0];
            if (Directory.Exists(mainDir)) ClearDirectory(mainDir);
            Directory.CreateDirectory(mainDir);

            for (var i = 0; i < activatonFunctions.Length; i++)
            {
                var activationFunction = activatonFunctions[i];

                Console.WriteLine($"Running experiment for {activationFunction}");

                var trainingOptions = new NeuralNetworkOptions(
                    options.LearningRate,
                    options.Momentum,
                    options.ErrorThreshold,
                    options.Sizes,
                    options.TrainingPath,
                    options.ValidationPath,
                    options.TestPath,
                    options.MaxEpochs,
                    options.IsVerbose,
                    options.BatchSize,
                    activationFunction,
                    options.InitialWeightsRange,
                    true,
                    options.NormalizeInput,
                    options.IsEncoder,
                    options.Lambda, options.TakeBest
                    );

                #region dump used params
                //lel
                var dumpling = JsonConvert.SerializeObject(options, Formatting.Indented);
                File.WriteAllText(logPath + ".log", dumpling);
                #endregion

                var trainingResponses = new TrainingResult[repetitions];

                var runLogPath = logPath + "/" + activationFunction;
                Directory.CreateDirectory(runLogPath);

                //gather data
                for (var j = 0; j < repetitions; j++)
                {
                    var trainingResponse = MnistTrainer.TrainOnMnist(trainingOptions);
                    trainingResponses[j] = trainingResponse;

                    File.WriteAllText($"{runLogPath}/{activationFunction}_{j}.json", trainingResponse.NeuralNetwork.ToJson());
                }

                var fileName = logPath + "_" + activationFunction;

                //log data
                var path = fileName + ".csv";

                //File.Create(path);

                var log = new StringBuilder("sep=|");
                log.AppendLine();
                log.Append("epoch");
                for (var j = 0; j < trainingResponses.Length; j++)
                {
                    log.Append("|evaluation_" + j + "|error_" + j);
                }
                log.AppendLine();
                for (var j = 0; j < trainingResponses[0].Epochs; j++)
                {
                    log.Append(j);
                    for (var n = 0; n < trainingResponses.Length; n++)
                    {
                        var result = trainingResponses[n];
                        log.Append("|" + result.Evaluations[j].Percentage + "|" + result.EpochErrors[j]);
                    }
                    log.AppendLine();
                }
                File.WriteAllText(path, log.ToString());

                #region dump plot
                ExperimentVisualization.GenerateEvaluationPlot(trainingResponses, fileName, fileName);
                ExperimentVisualization.GenerateErrorPlot(trainingResponses, logPath + "_error_" + activationFunction, fileName);
                #endregion
            }
        }

        public static void RunMomentumExperiment(
            double[] momentums,
            NeuralNetworkOptions options,
            int repetitions,
            string logPath
            )
        {
            //disable early learning end
            options.ErrorThreshold = 0;
            var isVerbose = options.IsVerbose;

            var mainDir = logPath.Split('/')[0];
            if (Directory.Exists(mainDir)) ClearDirectory(mainDir);
            Directory.CreateDirectory(mainDir);

            for (var i = 0; i < momentums.Length; i++)
            {
                var momentum = momentums[i];

                Console.WriteLine($"Running experiment for {momentum}");

                var trainingOptions = new NeuralNetworkOptions(
                    options.LearningRate,
                    momentum,
                    options.ErrorThreshold,
                    options.Sizes,
                    options.TrainingPath,
                    options.ValidationPath,
                    options.TestPath,
                    options.MaxEpochs,
                    options.IsVerbose,
                    options.BatchSize,
                    options.ActivationFunction,
                    options.InitialWeightsRange,
                    true,
                    options.NormalizeInput,
                    options.IsEncoder,
                    options.Lambda, options.TakeBest
                    );

                #region dump used params
                //lel
                var dumpling = JsonConvert.SerializeObject(options, Formatting.Indented);
                File.WriteAllText(logPath + ".log", dumpling);
                #endregion

                var trainingResponses = new TrainingResult[repetitions];

                var runLogPath = logPath + "/" + momentum;
                Directory.CreateDirectory(runLogPath);

                //gather data
                for (var j = 0; j < repetitions; j++)
                {
                    var trainingResponse = MnistTrainer.TrainOnMnist(trainingOptions);
                    trainingResponses[j] = trainingResponse;

                    File.WriteAllText($"{runLogPath}/{momentum}_{j}.json", trainingResponse.NeuralNetwork.ToJson());
                }

                var fileName = logPath + "_" + momentum;

                //log data
                var path = fileName + ".csv";

                //File.Create(path);

                var log = new StringBuilder("sep=|");
                log.AppendLine();
                log.Append("epoch");
                for (var j = 0; j < trainingResponses.Length; j++)
                {
                    log.Append("|evaluation_" + j + "|error_" + j);
                }
                log.AppendLine();
                for (var j = 0; j < trainingResponses[0].Epochs; j++)
                {
                    log.Append(j);
                    for (var n = 0; n < trainingResponses.Length; n++)
                    {
                        var result = trainingResponses[n];
                        log.Append("|" + result.Evaluations[j].Percentage + "|" + result.EpochErrors[j]);
                    }
                    log.AppendLine();
                }
                File.WriteAllText(path, log.ToString());

                #region dump plot
                ExperimentVisualization.GenerateEvaluationPlot(trainingResponses, fileName, fileName);
                ExperimentVisualization.GenerateErrorPlot(trainingResponses, logPath + "_error_" + momentum, fileName);
                #endregion
            }
        }

        public static void RunSizeExperiment(
            int[][] sizesArray,
            NeuralNetworkOptions options,
            int repetitions,
            string logPath
            )
        {
            //disable early learning end
            options.ErrorThreshold = 0;
            var isVerbose = options.IsVerbose;

            var mainDir = logPath.Split('/')[0];
            if (Directory.Exists(mainDir)) ClearDirectory(mainDir);
            Directory.CreateDirectory(mainDir);

            for (var i = 0; i < sizesArray.Length; i++)
            {
                var sizes = sizesArray[i];

                var serializedSizes = JsonConvert.SerializeObject(sizes);

                Console.WriteLine($"Running experiment for {serializedSizes}");

                var trainingOptions = new NeuralNetworkOptions(
                    options.LearningRate,
                    options.Momentum,
                    options.ErrorThreshold,
                    sizes,
                    options.TrainingPath,
                    options.ValidationPath,
                    options.TestPath,
                    options.MaxEpochs,
                    options.IsVerbose,
                    options.BatchSize,
                    options.ActivationFunction,
                    options.InitialWeightsRange,
                    true,
                    options.NormalizeInput,
                    options.IsEncoder,
                    options.Lambda, options.TakeBest
                    );

                #region dump used params
                //lel
                var dumpling = JsonConvert.SerializeObject(options, Formatting.Indented);
                File.WriteAllText(logPath + ".log", dumpling);
                #endregion

                var trainingResponses = new TrainingResult[repetitions];

                var runLogPath = logPath + "/" + serializedSizes;
                Directory.CreateDirectory(runLogPath);

                //gather data
                for (var j = 0; j < repetitions; j++)
                {
                    var trainingResponse = MnistTrainer.TrainOnMnist(trainingOptions);
                    trainingResponses[j] = trainingResponse;

                    File.WriteAllText($"{runLogPath}/{serializedSizes}_{j}.json", trainingResponse.NeuralNetwork.ToJson());
                }

                var fileName = logPath + "_" + serializedSizes;

                //log data
                var path = fileName + ".csv";

                //File.Create(path);

                var log = new StringBuilder("sep=|");
                log.AppendLine();
                log.Append("epoch");
                for (var j = 0; j < trainingResponses.Length; j++)
                {
                    log.Append("|evaluation_" + j + "|error_" + j);
                }
                log.AppendLine();
                for (var j = 0; j < trainingResponses[0].Epochs; j++)
                {
                    log.Append(j);
                    for (var n = 0; n < trainingResponses.Length; n++)
                    {
                        var result = trainingResponses[n];
                        log.Append("|" + result.Evaluations[j].Percentage + "|" + result.EpochErrors[j]);
                    }
                    log.AppendLine();
                }
                File.WriteAllText(path, log.ToString());

                #region dump plot
                if (!options.IsEncoder)
                {
                    ExperimentVisualization.GenerateEvaluationPlot(trainingResponses, fileName, fileName);
                }
                ExperimentVisualization.GenerateErrorPlot(trainingResponses, fileName, fileName);
                #endregion
            }
        }

        private static void ClearDirectory(string mainDir)
        {
            var di = new DirectoryInfo(mainDir);

            foreach (var file in di.GetFiles())
            {
                file.Delete();
            }
            foreach (var dir in di.GetDirectories())
            {
                dir.Delete(true);
            }
        }
    }
}