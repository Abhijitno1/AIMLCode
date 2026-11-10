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
		[VectorType(1, 256)] // batch size x sequence length
		[ColumnName("input_ids")]
		public long[] InputIds { get; set; }

		[VectorType(1, 256)]
		[ColumnName("attention_mask")]
		public long[] AttentionMask { get; set; }

		[VectorType(1, 256)]
		[ColumnName("token_type_ids")]
		public long[] TokenTypeIds { get; set; }
	}

	public class EmbeddingOutput
	{
		[ColumnName("last_hidden_state")]
		public float[] Vector { get; set; }
	}
}