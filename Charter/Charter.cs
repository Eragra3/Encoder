﻿using System.Collections.Generic;
using System.Windows.Forms.DataVisualization.Charting;

namespace Charter
{
    public static class Charter
    {
        public static void GeneratePlot(IList<DataPoint>[] seriesArray, string path, string title)
        {
            using (var ch = new Chart())
            {
                ch.ChartAreas.Add(new ChartArea());
                for (var i = 0; i < seriesArray.Length; i++)
                {
                    var series = seriesArray[i];
                    var s = new Series();
                    foreach (var pnt in series) s.Points.Add(pnt);
                    ch.Series.Add(s);
                    ch.Series[i].ChartType = SeriesChartType.Line;
                }
                ch.ChartAreas[0].AxisX.Minimum = 0;
                ch.Width = 500;
                ch.Titles.Add(new Title(title, Docking.Top));

                ch.SaveImage(path, ChartImageFormat.Png);
            }
        }

        public static void GeneratePlot(IList<DataPoint>[] seriesArray, string path, string title, int min, int max, int interval)
        {
            using (var ch = new Chart())
            {
                ch.ChartAreas.Add(new ChartArea());
                for (var i = 0; i < seriesArray.Length; i++)
                {
                    var series = seriesArray[i];
                    var s = new Series();
                    foreach (var pnt in series) s.Points.Add(pnt);
                    ch.Series.Add(s);
                    ch.Series[i].ChartType = SeriesChartType.Line;
                }
                ch.ChartAreas[0].AxisY.Minimum = min;
                ch.ChartAreas[0].AxisY.Maximum = max;
                ch.ChartAreas[0].AxisY.Interval = interval;
                ch.ChartAreas[0].AxisX.Minimum = 0;
                ch.Width = 500;
                ch.Titles.Add(new Title(title, Docking.Top));

                ch.SaveImage(path, ChartImageFormat.Png);
            }
        }
    }
}
