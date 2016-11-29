namespace Encoder.Serialization
{
    public class MatrixSerializationModel
    {
        public double[] Storage { get; set; }

        public int ColumnCount { get; set; }

        public int RowCount { get; set; }

        public bool IsDense { get; set; }

        public MatrixSerializationModel() { }
    }
}
