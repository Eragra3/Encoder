﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Encoder.Network
{
    public class NeuralNetworkOptions
    {
        public double LearningRate { get; set; }

        public double Momentum { get; set; }

        public double ErrorThreshold { get; set; }

        public int MaxEpochs { get; set; }

        public int[] Sizes { get; set; }

        public string TrainingPath { get; set; }

        public string ValidationPath { get; set; }

        public string TestPath { get; set; }

        public bool IsVerbose { get; set; }

        public int BatchSize { get; set; }

        public ActivationFunction ActivationFunction { get; set; }

        public double InitialWeightsRange { get; set; }

        public bool LogData { get; set; }

        public bool NormalizeInput { get; set; }

        public NeuralNetworkOptions(
            double learningRate,
            double momentum,
            double errorThreshold,
            int[] sizes,
            string trainingPath,
            string validationPath,
            string testPath,
            int maxEpochs,
            bool isVerbose,
            int batchSize,
            ActivationFunction activationFunction,
            double initialWeightsRange,
            bool logData,
            bool normalizeInput
            )
        {
            LearningRate = learningRate;
            Momentum = momentum;
            ErrorThreshold = errorThreshold;
            Sizes = sizes;
            TrainingPath = trainingPath;
            ValidationPath = validationPath;
            TestPath = testPath;
            MaxEpochs = maxEpochs;
            IsVerbose = isVerbose;
            BatchSize = batchSize;
            ActivationFunction = activationFunction;
            InitialWeightsRange = initialWeightsRange;
            LogData = logData;
            NormalizeInput = normalizeInput;
        }
    }
}