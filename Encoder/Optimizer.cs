using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;

namespace Encoder
{
    public static class Optimizer
    {
        public static bool TryUseNativeMKL()
        {
            return Control.TryUseNativeMKL();
        }
    }
}
