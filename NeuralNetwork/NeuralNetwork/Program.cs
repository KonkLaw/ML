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
					double[] networkOutput = network.ComputeOutput(_trainset[sampleIndex].Item1);
					for (int iteminSampleIndex = 0; iteminSampleIndex < networkOutput.Length; iteminSampleIndex++)
					{
						iterationError[iteminSampleIndex] = networkOutput[iteminSampleIndex] - _trainset[sampleIndex].Item2[iteminSampleIndex];
					}
					error += config.E_ErroFunr(iterationError);
					network.BackwardPass(config, networkOutput, iterationError);
				}

				// check
				error /= _trainset.Length;
				Console.WriteLine(error);
				if (error < config.Error)
					break;
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

		public void BackwardPass(Config config, double[] networkOutput, double[] iterationError)
		{
			// backprop

			// on putput

			// calc error:

			double[] errArgsOnOutput = new double[_outputLayer.Neurons.Length];


			//TODO: check this sum
			for (int prevIndex = 0; prevIndex < errArgsOnOutput.Length; prevIndex++)
			{
				for (int outputIndex = 0; outputIndex < _outputLayer.Neurons.Length; outputIndex++)
				{
					errArgsOnOutput[prevIndex] +=
						-1 * //!!!! TODO
		_outputLayer.Neurons[outputIndex].Weghts[prevIndex] *
		iterationError[outputIndex] /* dE / dSum_prevIndex (Sum_onStep) == error */
		* _outputLayer.Neurons[outputIndex].Deactivate(_outputLayer.Neurons[outputIndex].LastArg) /* df/dSum (Sum_onStep) */;
				}
			}

			// correct weights

			for (int neuronIndex = 0; neuronIndex < _outputLayer.Neurons.Length; neuronIndex++)
			{
				for (int weightIndex = 0; weightIndex < _outputLayer.Neurons[neuronIndex].Weghts.Length; weightIndex++)
				{
					// w_new = w_old + dw * delta
					_outputLayer.Neurons[neuronIndex].Weghts[weightIndex] +=
		-config.LearningRate *
		iterationError[neuronIndex] *
		_outputLayer.Neurons[neuronIndex].Deactivate(_outputLayer.Neurons[neuronIndex].LastArg) /* df/dSum (Sum_onStep) */
		* 

					;
				}
			}


			// correct weights


			//double[] dPrevValues = new double[_hiddenLayer.Neurons.Length];
			for (int prevNeuronIndex = 0; prevNeuronIndex < dPrevValues.Length; prevNeuronIndex++)
			{
				for (int j = 0; j < _outputLayer.Neurons.Length; j++)
				{
					dPrevValues[prevNeuronIndex] += iterationError[j] * _outputLayer.Neurons[i].;
				}
			}


			// iterationError => error on arg
			double[] dw = dF(output);
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

		public double LastArg;

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
			LastArg = sum;
			return ActFun(LastArg);
		}

		private double ActFun(double arg)
		{
			return Math.Pow(1 + Math.Exp(-arg), -1);
		}

		public double Deactivate(double arg)
		{
			// TODO:
			// thereticaly f'(arg) = ActFun'[arg]
			// but 
			return ActFun(arg) * (1 - ActFun(arg));
		}
	}
}
