using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;


namespace MachinelearningClass.HierarchyEmbedder
{

	public class TextData
	{
		[VectorType(1, 256)] // Adjust based on token length limit
		public long[] InputIds { get; set; }

		[VectorType(1, 256)]
		public long[] AttentionMask { get; set; }
	}

	public class EmbeddingOutput
	{
		[ColumnName("output_embeddings")]
		public float[] Vector { get; set; }
	}
}