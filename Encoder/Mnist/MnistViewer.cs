using System;
using System.Drawing;
using System.Globalization;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace Encoder.Mnist
{
    public static class MnistViewer
    {
        public static string ToMatrix(Vector<double> model, int width)
        {
            var sb = new StringBuilder();
            const string separator = "|";

            for (var i = 0; i < model.Count; i++)
            {
                sb.Append(Math.Floor(model[i] * 255).ToString(CultureInfo.InvariantCulture).PadRight(3, ' '));

                sb.Append((i + 1) % width == 0 ? "\n" : separator);
            }

            return sb.ToString();
        }

        public static Image ToImage(Vector<double> values, int width)
        {
            var image = new Bitmap(width, values.Count / width);

            for (var i = 0; i < values.Count; i++)
            {
                var x = i % width;
                var y = i / width;
                var c = values[i];
                c *= 255;
                var cInt = (int)c;
                var color = Color.FromArgb(cInt, cInt, cInt);
                image.SetPixel(x, y, color);
            }

            return image;
        }

        public static string Print(Vector<double> model, int width)
        {
            var sb = new StringBuilder();

            for (var i = 0; i < model.Count; i++)
            {
                if (model[i] > 0.75) sb.Append("@");
                else if (model[i] > 0.2) sb.Append("+");
                else sb.Append(" ");

                if ((i + 1) % width == 0) sb.Append("\n");
            }

            return sb.ToString();
        }

        public static string Dump(MnistModel image)
        {
            var sb = new StringBuilder();

            sb.Append(Print(image.Values, image.Width));
            sb.AppendLine($"Filename - {image.FileName}");
            sb.AppendLine($"Label - {image.Label}");
            sb.AppendLine($"Width - {image.Width}");
            sb.Append($"Height - {image.Height}");

            return sb.ToString();
        }
    }
}
