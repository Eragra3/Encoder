using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;

namespace Encoder.Network
{
    public abstract class Layer
    {
        public abstract Vector<double> Feedforward(Vector<double> input);
    }
}
