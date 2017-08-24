using System;

namespace NeuralNetwork
{
	internal class Config
	{
		public double Error { get; }

		public Config(double error)
		{
			Error = error;
		}

		internal double CalcItetaionError(double[] iterationError)
		{
			double error = 0;
			for (int i = 0; i < iterationError.Length; i++)
			{
				error += Math.Pow(iterationError[i], 2);
			}
			return error / iterationError.Length;
		}
	}
}