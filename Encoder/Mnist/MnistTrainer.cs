using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encoder.Network;

namespace Encoder.Mnist
{
    public static class MnistTrainer
    {
        private static MnistModel[] _trainingSet;
        private static string _trainingSetPath;
        private static MnistModel[] _testSet;
        private static string _testSetPath;
        private static MnistModel[] _validationSet;
        private static string _validationSetPath;

        public static TrainingResult TrainOnMnist(NeuralNetworkOptions options)
        {
            var isVerbose = options.IsVerbose;
            var normalize = options.NormalizeInput;

            var mlp = new NeuralNetwork(
                options.ActivationFunction,
                options.NormalStDeviation,
                options.Sizes);
            if (_trainingSetPath != options.TrainingPath || _trainingSet == null)
            {
                _trainingSet = MnistParser.ReadAll(options.TrainingPath, normalize);
                _trainingSetPath = options.TrainingPath;
            }
            var trainingSet = _trainingSet;
            if (_testSetPath != options.TestPath || _testSet == null)
            {
                _testSet = MnistParser.ReadAll(options.TestPath, normalize);
                _testSetPath = options.TestPath;
            }
            var testSet = _testSet;
            if (_validationSetPath != options.ValidationPath || _validationSet == null)
            {
                _validationSet = MnistParser.ReadAll(options.ValidationPath, normalize);
                _validationSetPath = options.ValidationPath;
            }
            var validationSet = _validationSet;

            var trainingModel = new TrainingModel
            {
                MaxEpochs = options.MaxEpochs,
                ErrorThreshold = options.ErrorThreshold,
                ValidationSet = validationSet,
                TrainingSet = trainingSet,
                TestSet = testSet,
                IsVerbose = isVerbose,
                BatchSize = options.BatchSize,
                LearningRate = options.LearningRate,
                Momentum = options.Momentum,
                EvaluateOnEachEpoch = options.EvaluateOnEachEpoch
            };

            var trainingResult = mlp.Train(trainingModel);

            return trainingResult;
        }
    }
}
