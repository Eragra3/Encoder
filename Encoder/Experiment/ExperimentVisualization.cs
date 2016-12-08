using System.Collections.Generic;
using System.Diagnostics;
using System.Windows.Forms.DataVisualization.Charting;
using Encoder.Network;

namespace Encoder.Experiment
{
    public static class ExperimentVisualization
    {
        public static void GenerateEvaluationPlot(
            TrainingResult trainingResult,
            string path,
            string title)
        {
            var series = new IList<DataPoint>[1];
            path += ".png";

            title += " - evaluation";

            series[0] = new List<DataPoint>(trainingResult.Evaluations.Length);
            for (var epoch = 0; epoch < trainingResult.Evaluations.Length; epoch++)
            {
                var evaluation = trainingResult.Evaluations[epoch];
                var dataPoint = new DataPoint(epoch, evaluation.Percentage);
                series[0].Add(dataPoint);
            }

            Charter.Charter.GeneratePlot(series, path, title, 0, 100, 10);
        }

        public static void GenerateEvaluationPlot(
            TrainingResult[] trainingResults,
            string path,
            string title)
        {
            title += " - evaluation";

            var series = new IList<DataPoint>[trainingResults.Length];
            path += ".png";

            for (var index = 0; index < trainingResults.Length; index++)
            {
                var trainingResult = trainingResults[index];
                series[index] = new List<DataPoint>(trainingResult.Evaluations.Length);
                var oneSeries = series[index];
                for (var epoch = 0; epoch < trainingResult.Evaluations.Length; epoch++)
                {
                    var evaluation = trainingResult.Evaluations[epoch];
                    var dataPoint = new DataPoint(epoch, evaluation.Percentage);
                    oneSeries.Add(dataPoint);
                }
            }

            Charter.Charter.GeneratePlot(series, path, title, 0, 100, 10);
        }

        public static void GenerateErrorPlot(
            TrainingResult trainingResult,
            string path,
            string title)
        {
            var series = new IList<DataPoint>[1];
            path += ".png";

            title += " - error";

            series[0] = new List<DataPoint>(trainingResult.EpochErrors.Length);
            for (var epoch = 0; epoch < trainingResult.EpochErrors.Length; epoch++)
            {
                var error = trainingResult.EpochErrors[epoch];
                var dataPoint = new DataPoint(epoch, error);
                series[0].Add(dataPoint);
            }

            Charter.Charter.GeneratePlot(series, path, title);
        }

        public static void GenerateErrorPlot(TrainingResult[] trainingResults, string path, string title)
        {
            var series = new IList<DataPoint>[trainingResults.Length];
            path += ".png";

            title += " - error";

            for (var index = 0; index < trainingResults.Length; index++)
            {
                var trainingResult = trainingResults[index];
                series[index] = new List<DataPoint>(trainingResult.EpochErrors.Length);
                var oneSeries = series[index];
                for (var epoch = 0; epoch < trainingResult.EpochErrors.Length; epoch++)
                {
                    var error = trainingResult.EpochErrors[epoch];
                    var dataPoint = new DataPoint(epoch, error);
                    oneSeries.Add(dataPoint);
                }
            }

            Charter.Charter.GeneratePlot(series, path, title);
        }

    }
}
