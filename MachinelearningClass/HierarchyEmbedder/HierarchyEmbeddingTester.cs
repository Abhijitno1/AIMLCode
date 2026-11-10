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

			var vocabPath = @"D:\Pract\Mine\Misc\TreeNodeSearchModel\hierarchical_transformer_model_onnx\vocab.txt";
			var tokenizer = BertTokenizer.Create(vocabPath, new BertOptions());
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
