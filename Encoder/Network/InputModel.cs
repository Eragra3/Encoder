using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace Encoder.Network
{
    public class InputModel
    {
        public Vector<double> Values { get; set; }

        public Vector<double> ExpectedSolution { get; set; }

        public int Label { get; set; }
    }
}
