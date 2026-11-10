using Microsoft.ML.Tokenizers;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass.HierarchyEmbedder
{
	public class HierarchyEmbeddingEngine
	{
		private readonly MLContext _mlContext;
		private readonly ITransformer _onnxModel;
		private readonly Tokenizer _tokenizer;

				public HierarchyEmbeddingEngine(string onnxModelPath, Tokenizer tokenizer)
		{
			_mlContext = new MLContext();
			_tokenizer = tokenizer;

			// Define the ML.NET Pipeline mapping C# inputs to ONNX model inputs/outputs
			var pipeline = _mlContext.Transforms.ApplyOnnxModel(
				outputColumnNames: new[] { "last_hidden_state" },
				inputColumnNames: new[] { "input_ids", "attention_mask", "token_type_ids" },
				modelFile: onnxModelPath,
				shapeDictionary: new Dictionary<string, int[]>
				{
					["input_ids"] = new[] { 1, 256 },
					["attention_mask"] = new[] { 1, 256 },
					["token_type_ids"] = new[] { 1, 256 }
				},
				gpuDeviceId: null,
				fallbackToCpu: false,
				recursionLimit: 100
			);

			// Fit an empty data view to instantiate the transformer wrapper
			var emptyData = _mlContext.Data.LoadFromEnumerable(new List<TextData>());
			_onnxModel = pipeline.Fit(emptyData);
		}

		public float[] GenerateEmbedding(string text)
		{
			// Encode text to token IDs using the ML.NET Tokenizer
			var tokenIds = _tokenizer.EncodeToIds(text).ToArray();
			int maxLength = 256;
			long[] inputIds = new long[maxLength];
			long[] attentionMask = new long[maxLength];
			int len = Math.Min(tokenIds.Length, maxLength);
			for (int i = 0; i < len; i++)
			{
				inputIds[i] = tokenIds[i];
				attentionMask[i] = 1;
			}
			// Remaining positions stay 0 (padding)
			return GenerateVector(inputIds, attentionMask);
		}

		public float[] GenerateVector(long[] inputIds, long[] attentionMask)
		{
			var predictionEngine = _mlContext.Model.CreatePredictionEngine<TextData, EmbeddingOutput>(_onnxModel);

			var prediction = predictionEngine.Predict(new TextData
			{
				InputIds = inputIds,
				AttentionMask = attentionMask,
				TokenTypeIds = new long[inputIds.Length]
			});

			// The ONNX model outputs token-level embeddings: [seq_len, dim]
			// We must perform Mean Pooling to get a single sentence embedding: [dim]
			float[] rawVector = prediction.Vector;
			int seqLen = inputIds.Length;
			int dim = rawVector.Length / seqLen;

			float[] pooledVector = new float[dim];
			int nonPaddingCount = 0;

			for (int i = 0; i < seqLen; i++)
			{
				// Only include tokens that are not padding (attentionMask[i] == 1)
				if (attentionMask[i] == 1)
				{
					nonPaddingCount++;
					for (int d = 0; d < dim; d++)
					{
						pooledVector[d] += rawVector[i * dim + d];
					}
				}
			}

			if (nonPaddingCount > 0)
			{
				for (int d = 0; d < dim; d++)
				{
					pooledVector[d] /= nonPaddingCount;
				}
			}

			return pooledVector;
		}
	}
}
