using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encoder.Network;

namespace Encoder.Mnist
{
    public class MnistModel : InputModel
    {
        public int Height { get; set; }
        public int Width { get; set; }
        public string FileName { get; set; }
    }
}
