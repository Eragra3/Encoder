using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace Encoder.Mnist
{
    public static class MnistViewer
    {
        public static string ToMatrix(Vector<double> model, int width)
        {
            var sb = new StringBuilder();
            const string separator = "|";

            for (int i = 0; i < model.Count; i++)
            {
                sb.Append(Math.Floor(model[i] * 255).ToString(CultureInfo.InvariantCulture).PadRight(3, ' '));

                sb.Append((i + 1) % width == 0 ? "\n" : separator);
            }

            return sb.ToString();
        }

        public static string Print(Vector<double> model, int width)
        {
            var sb = new StringBuilder();

            for (int i = 0; i < model.Count; i++)
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
