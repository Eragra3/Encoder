using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Encoder.Mnist
{
    public class MnistParser
    {
        public static readonly char[] FileSeparators = { '/', '\\' };

        public static MnistModel ReadImage(string path, bool normalize, bool trainAsEncoder)
        {
            var bitmap = new Bitmap(path);

            var fileName = path.Substring(path.LastIndexOfAny(FileSeparators) + 1);
            var label = int.Parse(fileName[0].ToString());

            var values = new DenseVector(bitmap.Height * bitmap.Width);

            for (var i = 0; i < bitmap.Height; i++)
            {
                for (var j = 0; j < bitmap.Width; j++)
                {
                    {
                        var color = bitmap.GetPixel(j, i);

                        var gray = (int)(color.R * 0.2126 + color.G * 0.7152 + color.B * 0.0722);

                        values[j + i * bitmap.Width] =1 -  gray / 255.0;
                    }
                }
            }

            if (normalize)
            {
                var max = values.Maximum();
                max /= 2;
                values.MapInplace(v => v - max);
            }

            //values.CoerceZero(0.005);
            //values.MapInplace(v => v > 0.995 ? 1 : v);

            Vector solution;
            if (trainAsEncoder)
            {
                solution = values;
            }
            else
            {
                solution = new DenseVector(10)
                {
                    [label] = 1.0
                };
            }

            var image = new MnistModel
            {
                Width = bitmap.Width,
                Height = bitmap.Height,
                Values = values,
                FileName = fileName,
                Label = label,
                ExpectedSolution = solution
            };

            return image;
        }

        public static MnistModel[] ReadAll(string pathToDirectory, bool normalize, bool trainAsEncoder)
        {
            var directoryInfo = new DirectoryInfo(pathToDirectory);

            var files = directoryInfo.GetFiles("*.png");
            var count = files.Length;
            var models = new MnistModel[count];

            for (var i = 0; i < files.Length; i++)
            {
                models[i] = ReadImage(files[i].FullName, normalize, trainAsEncoder);
            }

            return models;
        }
    }
}
