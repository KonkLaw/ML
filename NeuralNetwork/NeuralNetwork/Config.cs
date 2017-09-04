using System;
using System.Diagnostics;

namespace NeuralNetwork
{
	internal static class Config
	{
        public static double Error { get; } = 0.001d;

        public static readonly double LearningRate = 0.1d;

		internal static double E_ErroFunn(double[] result, double[] target)
		{
            Debug.Assert(result.Length == target.Length);
            double error = 0;
            for (int i = 0; i < result.Length; i++)
            {
                double dif = (target[i] - result[i]);
                error += dif * dif;
            }
			return error / 2;
		}

        internal static double E_ErroFunnDev(double[] result, double[] target, int derivativeComponent)
        {
            return -(target[derivativeComponent] - result[derivativeComponent]);
        }
    }
}