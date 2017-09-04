using System;
using System.Diagnostics;
using System.IO;
using System.Xml.Serialization;

namespace NeuralNetwork
{
    class Program
	{
		private static readonly (double[] Intput, double[] TargetOutput)[] Trainset =
		{
			(new double[]{ 0, 0 }, new double[]{ 0, 1 }),
			(new double[]{ 0, 1 }, new double[]{ 1, 0 }),
			(new double[]{ 1, 0 }, new double[]{ 1, 0 }),
			(new double[]{ 1, 1 }, new double[]{ 0, 1 })
		};

		static void Main()
		{
			var network = new Network();
			//var network = Network.Restore();
			Train(network);
			Console.WriteLine("End test");
			Test(network);
			//new Network().Save();
			Console.ReadKey();
		}

		static void Train(Network network)
		{
            int numStep = 0;
		    double edgeError;
		    do
		    {
		        double[] edgeErrors = new double[Trainset.Length];
		        for (int sampleIndex = 0; sampleIndex < Trainset.Length; sampleIndex++)
		        {
		            double[] networkOutput = network.ComputeOutput(Trainset[sampleIndex].Item1);
		            double[] target = Trainset[sampleIndex].Item2;
		            double error = Config.E_ErroFunn(networkOutput, target);
		            network.BackwardPass(networkOutput, target);
		            edgeErrors[sampleIndex] = error;
		        }

		        edgeError = edgeErrors.Avg();
		        numStep++;
		        Console.WriteLine($"StepNumber = {numStep} Error = {edgeError}");
		        //if (numStep % 1000 == 0)
		        {
		            //network.Show();
		        }
		    } while (edgeError > Config.Error);
		}

		private static void Test(Network network)
		{
            for (int i = 0; i < Trainset.Length; i++)
            {
                network.ComputeOutput(Trainset[i].Intput).WriteLine("Result");
                Trainset[i].TargetOutput.WriteLine("Target");
            }
        }
	}

	class Layer
	{
		protected Neuron[] Neurons;
		private readonly double[] _output;
        protected double[] LastIutput;

        public Layer(int inputsCount, int outputCount)
		{
			Neurons = new Neuron[outputCount];
			for (int i = 0; i < Neurons.Length; i++)
			{
				Neurons[i] = new Neuron(inputsCount);
			}
			_output = new double[outputCount];
		}

		public double[] CalcOutput(double[] input)
		{
            LastIutput = input;
            for (int i = 0; i < Neurons.Length; i++)
			{
				_output[i] = Neurons[i].Activate(input);
			}
			return _output;
		}

		public double[][] GetMatrix()
		{
			double[][] matr = new double[Neurons.Length][];
			for (int i = 0; i < matr.Length; i++)
			{
				matr[i] = Neurons[i].Weghts;
			}
			return matr;
		}

        internal void SetMatrix(double[][] hiddenLayerWeights)
		{
			Neurons = new Neuron[hiddenLayerWeights.Length];
			for (int i = 0; i < Neurons.Length; i++)
			{
				Neurons[i] = new Neuron(hiddenLayerWeights[i]);
			}
		}

        public void BackwardPass(double[] gradSums)
        {
            for (int neuronIndex = 0; neuronIndex < Neurons.Length; neuronIndex++)
            {
                Neuron neuron = Neurons[neuronIndex];
                double[] weights = Neurons[neuronIndex].Weghts;
                for (int weightIndex = 0; weightIndex < weights.Length; weightIndex++)
                {
                    // - rate * o_i * dEDS_j
                    weights[weightIndex] +=
                        -1 * Config.LearningRate * LastIutput[weightIndex]
                         * GetHiddenLocalGrad(neuronIndex, gradSums[neuronIndex]);
                }
            }
        }

        private double GetHiddenLocalGrad(int iNeuron, double gradSum)
        {
            Neuron neuron = Neurons[iNeuron];
            return neuron.DeactivateByValue(_output[iNeuron]) * gradSum;
        }

        internal void Show()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Show();
            }
        }
    }

    class OutputLayer : Layer
    {
        public OutputLayer(int inputsCount, int outputCount)
            : base(inputsCount, outputCount) { }

        public void BackwardPass1(double[] target, double[] networkOutput, out double[] gradSums)
        {
            double[] localGrads = new double[Neurons.Length];
            for (int neuronIndex = 0; neuronIndex < Neurons.Length; neuronIndex++)
            {
                Neuron neuron = Neurons[neuronIndex];
                localGrads[neuronIndex] =
                    Config.E_ErroFunnDev(networkOutput, target, neuronIndex)
                    * (neuron.DeactivateByValue(networkOutput[neuronIndex]));
            }

            //TODO: !! 1 !!
            gradSums = new double[LastIutput.Length];
            for (int neuronIndex = 0; neuronIndex < Neurons.Length; neuronIndex++)
            {
                for (int inputIndex = 0; inputIndex < gradSums.Length; inputIndex++)
                {
                    gradSums[inputIndex] +=
                        localGrads[neuronIndex] * Neurons[neuronIndex].Weghts[inputIndex];
                }
            }         

            for (int neuronIndex = 0; neuronIndex < Neurons.Length; neuronIndex++)
            {
                Neuron neuron = Neurons[neuronIndex];
                double[] weights = Neurons[neuronIndex].Weghts;
                for (int weightIndex = 0; weightIndex < weights.Length; weightIndex++)
                {
                    // - rate * o_i * dEDS_j
                    weights[weightIndex] +=
                        -1 * Config.LearningRate * LastIutput[weightIndex]
                         * localGrads[neuronIndex];
                }
            }
        }
    }

	class Network
	{
		private const string FileName = "data.xml";
		private readonly Layer _hiddenLayer;
		private readonly OutputLayer _outputLayer;

		public Network() : this(2, 4, 2) { }

		public Network(Layer hiddenLayer, OutputLayer outputLayer)
		{
			_hiddenLayer = hiddenLayer; _outputLayer = outputLayer;
		}

		public Network(int inputCount, int hiddenCount, int outputCount)
			: this(
                new Layer(inputCount, hiddenCount),
                new OutputLayer(hiddenCount, outputCount)) { }

		public double[] ComputeOutput(double[] input)
		{
			// TODO: place for cycle
			double[] t1 = _hiddenLayer.CalcOutput(input);
			double[] t2 = _outputLayer.CalcOutput(t1);
			return t2;
		}

		public void BackwardPass(double[] networkOutput, double[] target)
		{
            double[] localGrads;
            _outputLayer.BackwardPass1(target, networkOutput, out localGrads);
            _hiddenLayer.BackwardPass(localGrads);
        }

		public void Save()
		{
			var serializer = new XmlSerializer(typeof(SerializedNetwork));
			using (var writer = new StringWriter())
			{
				serializer.Serialize(writer, new SerializedNetwork
                {
					HiddenLayerWeights = _hiddenLayer.GetMatrix(),
					OutputLayerWeights = _outputLayer.GetMatrix(),
				});
				File.WriteAllText(FileName, writer.ToString());
			}
		}

		public static Network Restore()
		{
			var serializer = new XmlSerializer(typeof(SerializedNetwork));
			Network network;
			using (var reader = new StringReader(File.ReadAllText(FileName)))
			{
			    var data = (SerializedNetwork)serializer.Deserialize(reader);
				network = new Network(data.HiddenLayerWeights[0].Length, data.HiddenLayerWeights.Length, data.OutputLayerWeights.Length);
				network._hiddenLayer.SetMatrix(data.HiddenLayerWeights);
			    network._outputLayer.SetMatrix(data.OutputLayerWeights);
            }
			return network;
		}

        internal void Show()
        {
            _hiddenLayer.Show();
            _outputLayer.Show();
        }
    }

	class Neuron
	{
		private readonly double[] _inputWeghts;

		public double[] Weghts => _inputWeghts;

	    public Neuron(int inputWeghtsCount)
	    {
	        _inputWeghts = new double[inputWeghtsCount];
	        // Real problem was here!
	        for (int i = 0; i < _inputWeghts.Length; i++)
	        {
	            _inputWeghts[i] = Helpers.GetSmallRandom();
	        }
	    }

        public Neuron(double[] inputWeghts)
		{
            _inputWeghts = inputWeghts;
        }

        public double Activate(double[] input)
        {
            Debug.Assert(_inputWeghts.Length == input.Length);
            double sum = 0;
        	for (int i = 0; i < input.Length; i++)
        	{
        		sum += input[i] * Weghts[i];
        	}
            return ActFun(sum);
        }
        
        private static double ActFun(double arg)
        {
        	return Math.Pow(1 + Math.Exp(-arg), -1);
        }
        
        public double DeactivateByValue(double arg)
        {
        	// TODO:
        	// thereticaly f'(arg) = ActFun'[arg]
        	// but 
        	return arg * (1 - arg);
        }

        internal void Show()
        {
            for (int i = 0; i < Weghts.Length; i++)
            {
                Console.Write(Weghts[i] + " ");
            }
            Console.WriteLine();
        }
    }
}
