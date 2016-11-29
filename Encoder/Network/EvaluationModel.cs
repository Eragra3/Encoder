using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Encoder.Network
{
    public class EvaluationModel
    {
        public int All { get; set; }
        public int Correct { get; set; }

        public int Incorrect => All - Correct;
        public double Percentage => Math.Round((double)Correct / All * 100, 2);
    }
}
