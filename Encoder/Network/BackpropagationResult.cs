using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace Encoder.Network
{
    public class BackpropagationResult
    {
        public Vector<double>[] Biases { get; set; }

        public Matrix<double>[] Weights { get; set; }

        public Vector<double> Solution { get; set; }

        public Vector<double> Input { get; set; }
    }
}
