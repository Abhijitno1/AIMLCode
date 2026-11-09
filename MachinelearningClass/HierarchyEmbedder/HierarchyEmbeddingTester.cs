using AllMiniLmL6V2Sharp;
using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass.HierarchyEmbedder
{
	public class HierarchyEmbeddingTester
	{
		public static void Test()
		{

			// Use a concrete tokenizer implementation; here we use the Tiktoken tokenizer for the 'gpt-4o' model.
			var tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o");
			string onnxModelPath = @"D:\Pract\Mine\Misc\TreeNodeSearchModel\hierarchical_transformer_model_onnx\onnx\model.onnx";
			var engine = new HierarchyEmbeddingEngine(onnxModelPath, tokenizer);
			string texttobeMatched = "E:\\TestGround\\DirectoryMap\\Group 1\\Crescent Origins\\Achintya Mountain Figure\\Achintya in Bali.png";
			var texttobeMatchedV = engine.GenerateEmbedding(texttobeMatched).ToArray();
			Console.WriteLine("Enter text be matched");
			string inputText = "Achintya in Bali.png";
			var inputTextV = engine.GenerateEmbedding(inputText);
			Console.WriteLine(Common.CalculateCosineSimilarity(texttobeMatchedV, inputTextV));

		}
	}
}
