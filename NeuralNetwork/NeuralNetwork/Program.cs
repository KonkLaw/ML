using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace NeuralNetwork
{
	static class OutputHelper
	{
		public static void WriteLine(this double[] array)
		{
			foreach (double item in array)
			{
				Console.Write(item); Console.Write(" ");
			}
			Console.WriteLine();
		}
	}

	class Program
	{
		private static  (double[], double[])[] _trainset = new(double[], double[])[]
		{
			(new double[]{ 0, 0 }, new double[]{ 0, 1 }),
			(new double[]{ 0, 1 }, new double[]{ 1, 0 }),
			(new double[]{ 1, 0 }, new double[]{ 1, 0 }),
			(new double[]{ 1, 1 }, new double[]{ 0, 1 })
		};

		static void Main(string[] args)
		{
			var network = new Network();
			//var network = Network.Restore();
			var config = new Config(0.001);
			Train(network, config);
			Console.WriteLine("End test");
			Test(network);
			//new Network().Save();
			Console.ReadKey();
		}

		static void Train(Network network, Config config)
		{
			while (true)
			{
				// iterate
				double error = 0;
				double[] iterationError = new double[_trainset[0].Item1.Length];
				for (int sampleIndex = 0; sampleIndex < _trainset.Length; sampleIndex++)
				{
					double[] output = network.ComputeOutput(_trainset[sampleIndex].Item1);
					for (int iteminSampleIndex = 0; iteminSampleIndex < output.Length; iteminSampleIndex++)
					{
						iterationError[iteminSampleIndex] = output[iteminSampleIndex] - _trainset[sampleIndex].Item2[iteminSampleIndex];
					}
					error += config.CalcItetaionError(iterationError);
					network.BackwardPass(iterationError);
				}

				// check
				error /= _trainset.Length;
				Console.WriteLine(error);
				if (error < config.Error)
					break;

				// backprop
			}
		}

		private static void Test(Network network)
		{
			for (int i = 0; i < _trainset.Length; i++)
			{
				network.ComputeOutput(_trainset[i].Item1).WriteLine();
			}
		}
	}

	public class Data
	{
		public double[][] HiddenLayerWeights { get; set; }

		public double[][] OutputLayerWeights { get; set; }
	}

	class AcceptingLayer
	{
		private Neuron[] _neurons;
		private readonly double[] _output;

		internal Neuron[] Neurons => _neurons;

		public AcceptingLayer(int inputsCount, int outputCount)
		{
			_neurons = new Neuron[outputCount];
			for (int i = 0; i < _neurons.Length; i++)
			{
				_neurons[i] = new Neuron(new double[inputsCount]);
			}
			_output = new double[outputCount];
		}

		public double[] GetOutput(double[] input)
		{
			for (int i = 0; i < Neurons.Length; i++)
			{
				_output[i] = Neurons[i].Activate(input);
			}
			return _output;
		}

		public double[][] GetMatrix()
		{
			double[][] matr = new double[_neurons.Length][];
			for (int i = 0; i < matr.Length; i++)
			{
				matr[i] = _neurons[i].Weghts;
			}
			return matr;
		}

		internal void SetMatrix(double[][] hiddenLayerWeights)
		{
			_neurons = new Neuron[hiddenLayerWeights.Length];
			for (int i = 0; i < _neurons.Length; i++)
			{
				_neurons[i] = new Neuron(hiddenLayerWeights[i]);
			}
		}
	}

	class Network
	{
		private const string FileName = "data.xml";
		private readonly AcceptingLayer _hiddenLayer;
		private readonly AcceptingLayer _outputLayer;

		public Network() : this(2, 4, 2) { }

		public Network(AcceptingLayer hiddenLayer, AcceptingLayer outputLayer)
		{
			_hiddenLayer = hiddenLayer; _outputLayer = outputLayer;
		}

		public Network(int inputCount, int hiddenCount, int outputCount)
			: this(new AcceptingLayer(inputCount, hiddenCount), new AcceptingLayer(hiddenCount, outputCount)) { }

		public double[] ComputeOutput(double[] input)
		{
			// For should be placed here
			double[] t1 = _hiddenLayer.GetOutput(input);
			double[] t2 = _outputLayer.GetOutput(t1);
			return t2;
		}

		public void BackwardPass(double[] iterationError)
		{

		}

		public void Save()
		{
			var serializer = new XmlSerializer(typeof(Data));
			using (var writer = new StringWriter())
			{
				serializer.Serialize(writer, new Data
				{
					HiddenLayerWeights = _hiddenLayer.GetMatrix(),
					OutputLayerWeights = _outputLayer.GetMatrix(),
				});
				File.WriteAllText(FileName, writer.ToString());
			}
		}

		public static Network Restore()
		{
			var serializer = new XmlSerializer(typeof(Data));
			Network network;
			Data data;
			using (var reader = new StringReader(File.ReadAllText(FileName)))
			{
				data = (Data)serializer.Deserialize(reader);
				network = new Network(data.HiddenLayerWeights[0].Length, data.HiddenLayerWeights.Length, data.OutputLayerWeights.Length);
				network._hiddenLayer.SetMatrix(data.HiddenLayerWeights);
			}
			network._hiddenLayer.SetMatrix(data.HiddenLayerWeights);
			network._outputLayer.SetMatrix(data.OutputLayerWeights);
			return network;
		}
	}

	class Neuron
	{
		private double[] _weghts;

		public double[] Weghts { get => _weghts; }

		public Neuron(double[] weghts)
		{
			_weghts = weghts;
		}

		public double Activate(double[] input)
		{
			double sum = 0;
			for (int i = 0; i < input.Length; i++)
			{
				sum += input[i] * Weghts[i];
			}
			return Math.Pow(1 + Math.Exp(-sum), -1);
		}
	}
}
