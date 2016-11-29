using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.LinearAlgebra.Storage;
using Newtonsoft.Json;

namespace Encoder.Serialization
{
    public class VectorConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return typeof(DenseVector).IsAssignableFrom(objectType) ||
                typeof(SparseVector).IsAssignableFrom(objectType);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            var serializationModel = serializer.Deserialize<VectorSerializationModel>(reader);

            if (serializationModel.IsDense)
            {
                var storage = DenseVectorStorage<double>.OfEnumerable(
                    serializationModel.Storage
                    );
                var vector = new DenseVector(storage);
                return vector;
            }
            else
            {
                var storage = SparseVectorStorage<double>.OfEnumerable(
                    serializationModel.Storage
                    );
                var vector = new SparseVector(storage);
                return vector;
            }
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            var matrix = (Vector)value;
            var serializationModel = new VectorSerializationModel()
            {
                Storage = matrix.Storage.Enumerate().ToArray(),
                IsDense = matrix.Storage.IsDense
            };
            serializer.Serialize(writer, serializationModel);
        }
    }
}
