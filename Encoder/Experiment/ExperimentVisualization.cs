using System.Collections.Generic;
using System.Windows.Forms.DataVisualization.Charting;
using Encoder.Network;

namespace Encoder.Experiment
{
    public static class ExperimentVisualization
    {
        public static void GenerateEvaluationPlot(
            TrainingResult[] trainingResults,
            string path,
            int min,
            int max)
        {
            IList<DataPoint>[] series = new IList<DataPoint>[trainingResults.Length];
            path += ".png";

            for (int index = 0; index < trainingResults.Length; index++)
            {
                var trainingResult = trainingResults[index];
                series[index] = new List<DataPoint>(trainingResult.Evaluations.Length);
                var oneSeries = series[index];
                for (int epoch = 0; epoch < trainingResult.Evaluations.Length; epoch++)
                {
                    var evaluation = trainingResult.Evaluations[epoch];
                    var dataPoint = new DataPoint(epoch, evaluation.Percentage);
                    oneSeries.Add(dataPoint);
                }
            }

            Charter.Charter.GeneratePlot(series, path, min, max);
        }

        public static void GenerateErrorPlot(
            TrainingResult[] trainingResults,
            string path)
        {
            IList<DataPoint>[] series = new IList<DataPoint>[trainingResults.Length];
            path += ".png";

            for (int index = 0; index < trainingResults.Length; index++)
            {
                var trainingResult = trainingResults[index];
                series[index] = new List<DataPoint>(trainingResult.EpochErrors.Length);
                var oneSeries = series[index];
                for (int epoch = 0; epoch < trainingResult.EpochErrors.Length; epoch++)
                {
                    var error = trainingResult.EpochErrors[epoch];
                    var dataPoint = new DataPoint(epoch, error);
                    oneSeries.Add(dataPoint);
                }
            }

            Charter.Charter.GeneratePlot(series, path);
        }

    }
}
