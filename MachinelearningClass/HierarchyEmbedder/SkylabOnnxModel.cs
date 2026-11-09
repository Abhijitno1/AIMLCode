using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;


namespace MachinelearningClass.HierarchyEmbedder
{
	public class SkylabOnnxModel
	{
		public static void Run()
		{
			var session = new InferenceSession(
				@"D:\Pract\Mine\Misc\TreeNodeSearchModel\hierarchical_transformer_model_onnx\onnx\model.onnx");

			// Example tokenized input for 1 sentence
			// 1 x seq_len
			long[] inputIds = new long[]
			{
			101, 2027, 1055, 2463, 2007, 2419, 1045, 1037, 12480, 102, 0, 0
			};

			long[] attentionMask = new long[]
			{
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
			};

			long[] tokenTypeIds = new long[]
			{
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
			};

			var inputIdsTensor = new DenseTensor<long>(inputIds, new[] { 1, inputIds.Length });
			var attentionMaskTensor = new DenseTensor<long>(attentionMask, new[] { 1, attentionMask.Length });
			var tokenTypeIdsTensor = new DenseTensor<long>(tokenTypeIds, new[] { 1, tokenTypeIds.Length });

			var inputs = new List<NamedOnnxValue>
		{
			NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
			NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
			NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor)
		};

			using var results = session.Run(inputs);

			var embedding = results[0].AsTensor<float>();
			Console.WriteLine($"Embedding shape: {embedding.Dimensions[0]} x {embedding.Dimensions[1]}");
			// expected: 1 x 384
		}
	}
}
