using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encoder.Network;
using Newtonsoft.Json;

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
            var isEncoder = options.IsEncoder;

            #region dump used params
            //lel
            var dumpling = JsonConvert.SerializeObject(options, Formatting.Indented);
            File.WriteAllText("neural_network.log", dumpling);
            #endregion

            NeuralNetwork mlp;
            if (isEncoder)
            {
                mlp = new Network.Encoder(
                options.ActivationFunction,
                options.InitialWeightsRange,
                options.Sizes);
            }
            else
            {
                mlp = new NeuralNetwork(
                    options.ActivationFunction,
                    options.InitialWeightsRange,
                    options.Sizes);
            }

            if (_trainingSetPath != options.TrainingPath || _trainingSet == null)
            {
                _trainingSet = MnistParser.ReadAll(options.TrainingPath, normalize, isEncoder);
                _trainingSetPath = options.TrainingPath;
            }
            var trainingSet = _trainingSet;
            if (_testSetPath != options.TestPath || _testSet == null)
            {
                _testSet = MnistParser.ReadAll(options.TestPath, normalize, isEncoder);
                _testSetPath = options.TestPath;
            }
            var testSet = _testSet;
            if (_validationSetPath != options.ValidationPath || _validationSet == null)
            {
                _validationSet = MnistParser.ReadAll(options.ValidationPath, normalize, isEncoder);
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
                EvaluateOnEachEpoch = options.LogData,
                IsEncoder = options.IsEncoder
            };

            var trainingResult = mlp.Train(trainingModel);

            return trainingResult;
        }
    }
}
