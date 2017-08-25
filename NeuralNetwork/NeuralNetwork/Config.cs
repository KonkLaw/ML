using System;

namespace NeuralNetwork
{
	internal class Config
	{
		public double Error { get; }

		public readonly double LearningRate = 0.1d;

		public Config(double error)
		{
			Error = error;
		}

		internal double E_ErroFunr(double[] iterationError)
		{
			double error = 0;
			for (int i = 0; i < iterationError.Length; i++)
			{
				error += iterationError[i] * iterationError[i];
			}
			return error / iterationError.Length;
		}

		internal double dEdf_ErroFunrDer_i(double arg)
		{
			return arg;
		}


	}
}