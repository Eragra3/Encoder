using System;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Storage;
using Newtonsoft.Json;

namespace Encoder.Serialization
{
    public class MatrixConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return typeof(DenseMatrix).IsAssignableFrom(objectType) ||
                typeof(SparseMatrix).IsAssignableFrom(objectType);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            var serializationModel = serializer.Deserialize<MatrixSerializationModel>(reader);

            if (serializationModel.IsDense)
            {
                var storage = DenseColumnMajorMatrixStorage<double>.OfColumnMajorEnumerable(
                    serializationModel.RowCount,
                    serializationModel.ColumnCount,
                    serializationModel.Storage
                    );
                var matrix = new DenseMatrix(storage);
                return matrix;
            }
            else
            {
                var storage = SparseCompressedRowMatrixStorage<double>.OfColumnMajorList(
                    serializationModel.RowCount,
                    serializationModel.ColumnCount,
                    serializationModel.Storage
                    );
                var matrix = new SparseMatrix(storage);
                return matrix;
            }
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            var matrix = (Matrix)value;
            var serializationModel = new MatrixSerializationModel
            {
                RowCount = matrix.RowCount,
                ColumnCount = matrix.ColumnCount,
                Storage = matrix.Storage.ToColumnMajorArray(),
                IsDense = matrix.Storage.IsDense
            };
            serializer.Serialize(writer, serializationModel);
        }
    }
}
