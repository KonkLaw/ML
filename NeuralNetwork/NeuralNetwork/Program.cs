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
	class Program
	{
		static void Main(string[] args)
		{
			new Network();
			var network = Network.Restore();


			//var config = new Config(0.001);
			//Train(config);
			//var n = new Neuron(null);
			Console.ReadKey();
		}

		private static void Train(Config config)
		{
			while(true)
			{
				double error = 0;


				if (error < config.Error)
					return;

			}
		}
	}

	class AcceptingLayer
	{
		private readonly Neuron[] _neurons;
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
			Debug.Assert(input.Length == Neurons.Length);
			for (int i = 0; i < Neurons.Length; i++)
			{
				_output[i] = Neurons[i].Activate(input);
			}
			return _output;
		}
	}

	class Network
	{
		private const string FileName = "data.xml";
		private readonly AcceptingLayer _hiddenLayer;
		private readonly AcceptingLayer _outputLayer;

		public Network() : this(2, 4, 2) { }

		private Network(Neuron[] hiddenLayer, Neuron[] outputLayer)
		{
			_hiddenLayer = hiddenLayer;
			_outputLayer = 
			_hiddenLayer = new AcceptingLayer(intputCoutn, hiddenCount);
			_outputLayer = new AcceptingLayer(hiddenCount, outputCount);
		}

		public double[] ComputeOutput(double[] input)
		{
			// For should be placed here
			double[] t1 = _hiddenLayer.GetOutput(input);
			double[] t2 = _outputLayer.GetOutput(t1);
			return t2;
		}

		public void Save()
		{
			var serializer = new XmlSerializer(typeof(Data));
			using (var writer = new StringWriter())
			{
				serializer.Serialize(writer, new Data());
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
				network = new Network(data.InputCount, data.HiddenCount, data.OutputCount);
			}

			for (int i = 0; i < data.HiddenLayerWeights.Length; i++)
			{
				for (int j = 0; j < network._hiddenLayer.Neurons.Length; j++)
					network._hiddenLayer.Neurons[j].Weghts
			}


			
		}

		public class Data
		{
			public int InputCount { get; set; }
			public int HiddenCount { get; set; }
			public int OutputCount { get; set; }

			public double[][] HiddenLayerWeights { get; set; }

			public double[][] OutputLayerWeights { get; set; }
		}
	}


	class Neuron
	{
		private readonly double[] _weghts;

		public double[] Weghts => _weghts;

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
