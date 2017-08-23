using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	class Program
	{
		static void Main(string[] args)
		{
			Train();
			var n = new Neoron(null);


		}

		private static void Train()
		{
			double error = 0;
			for (int i = 0; i < 1; i++)
			{

			}
		}
	}


	class Neoron
	{
		private readonly double[] _weghts;

		public Neoron(double[] weghts)
		{
			_weghts = weghts;
		}

		public double CalNet(double[] inputs)
		{
			double sum = 0;
			for (int i = 0; i < inputs.Length; i++)
			{
				sum += inputs[i] * _weghts[i];
			}
			return Math.Pow(1 + Math.Exp(-sum), -1);
		}
	}
}
