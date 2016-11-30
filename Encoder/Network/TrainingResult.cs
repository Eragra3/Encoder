using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Encoder.Network
{
   public class TrainingResult
    {
        public NeuralNetwork NeuralNetwork { get; set; }

        public int Epochs { get; set; }

        public double[] EpochErrors { get; set; }

        public EvaluationModel[] Evaluations { get; set; }
    }
}
