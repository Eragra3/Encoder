using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encoder.Mnist;
using Encoder.Network;

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

            Directory.CreateDirectory(logPath.Split('/')[0]);

            for (var i = 0; i < learningRates.Length; i++)
            {
                var learningRate = learningRates[i];

                if (isVerbose) Console.WriteLine($"Running experiment for {learningRate}");

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
                    options.IsEncoder
                    );

                var trainingResponses = new TrainingResult[repetitions];

                //gather data
                for (var j = 0; j < repetitions; j++)
                {
                    var mlp = new NeuralNetwork(options.ActivationFunction, options.InitialWeightsRange, options.Sizes);

                    var trainingResponse = MnistTrainer.TrainOnMnist(trainingOptions);
                    trainingResponses[j] = trainingResponse;
                }

                //log data
                var path = logPath + "_" + learningRate + ".csv";

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
                ExperimentVisualization.GenerateEvaluationPlot(trainingResponses, logPath + "_" + learningRate);
                ExperimentVisualization.GenerateErrorPlot(trainingResponses, logPath + "_error_" + learningRate);
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

            Directory.CreateDirectory(logPath.Split('/')[0]);

            for (var i = 0; i < initialWeightsRanges.Length; i++)
            {
                var initialWeightsRange = initialWeightsRanges[i];

                if (isVerbose) Console.WriteLine($"Running experiment for {initialWeightsRange}");

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
                    options.IsEncoder
                    );

                var trainingResponses = new TrainingResult[repetitions];

                //gather data
                for (var j = 0; j < repetitions; j++)
                {
                    var trainingResponse = MnistTrainer.TrainOnMnist(trainingOptions);
                    trainingResponses[j] = trainingResponse;
                }

                //log data
                var path = logPath + "_" + initialWeightsRange + ".csv";

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
                ExperimentVisualization.GenerateEvaluationPlot(trainingResponses, logPath + "_" + initialWeightsRange);
                ExperimentVisualization.GenerateErrorPlot(trainingResponses, logPath + "_" + initialWeightsRange);
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

            Directory.CreateDirectory(logPath.Split('/')[0]);

            for (var i = 0; i < activatonFunctions.Length; i++)
            {
                var activationFunction = activatonFunctions[i];

                if (isVerbose) Console.WriteLine($"Running experiment for {activationFunction}");

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
                    options.IsEncoder
                    );

                var trainingResponses = new TrainingResult[repetitions];

                //gather data
                for (var j = 0; j < repetitions; j++)
                {
                    var trainingResponse = MnistTrainer.TrainOnMnist(trainingOptions);
                    trainingResponses[j] = trainingResponse;
                }

                //log data
                var path = logPath + "_" + activationFunction + ".csv";

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
                ExperimentVisualization.GenerateEvaluationPlot(trainingResponses, logPath + "_" + activationFunction);
                ExperimentVisualization.GenerateErrorPlot(trainingResponses, logPath + "_" + activationFunction);
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

            Directory.CreateDirectory(logPath.Split('/')[0]);

            for (var i = 0; i < momentums.Length; i++)
            {
                var momentum = momentums[i];

                if (isVerbose) Console.WriteLine($"Running experiment for {momentum}");

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
                    options.IsEncoder
                    );

                var trainingResponses = new TrainingResult[repetitions];

                //gather data
                for (var j = 0; j < repetitions; j++)
                {
                    var trainingResponse = MnistTrainer.TrainOnMnist(trainingOptions);
                    trainingResponses[j] = trainingResponse;
                }

                //log data
                var path = logPath + "_" + momentum + ".csv";

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
                ExperimentVisualization.GenerateEvaluationPlot(trainingResponses, logPath + "_" + momentum);
                ExperimentVisualization.GenerateErrorPlot(trainingResponses, logPath + "_" + momentum);
                #endregion
            }
        }
    }
}