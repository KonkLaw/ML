using System;
using System.Diagnostics;

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

		private static void Train(Network network)
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
		    } while (edgeError > Config.Error);
		}

		private static void Test(Network network)
		{
            for (int i = 0; i < Trainset.Length; i++)
            {
                network.ComputeOutput(Trainset[i].Intput).WriteLine("Result", 3);
                Trainset[i].TargetOutput.WriteLine("Target");
            }
        }
	}

	abstract class BaseLayer
	{
		protected readonly Neuron[] Neurons;
		protected readonly double[] LastOutput;

		public BaseLayer(int inputsCount, int outputCount)
		{
			Neurons = new Neuron[outputCount];
			for (int i = 0; i < Neurons.Length; i++)
			{
				Neurons[i] = new Neuron(inputsCount);
			}
			LastOutput = new double[outputCount];
		}

		public abstract double[] CalcOutput(double[] input);

		protected static void CorrectWeights(
			Neuron[] neurons, double[] prevOutput, double[] localGradients)
		{
			Debug.Assert(neurons.Length == localGradients.Length);
			Debug.Assert(neurons[0].Weghts.Length == prevOutput.Length);

			for (int neuronIndex = 0; neuronIndex < neurons.Length; neuronIndex++)
			{
				Neuron neuron = neurons[neuronIndex];
				double[] weights = neurons[neuronIndex].Weghts;
				for (int weightIndex = 0; weightIndex < weights.Length; weightIndex++)
				{
					weights[weightIndex] +=
						-1 * Config.LearningRate * prevOutput[weightIndex] * localGradients[neuronIndex];
				}
			}
		}

		protected static double[] CalcGradSums(
			double[] localGradients, Neuron[] neurons, int prevNeuronsCount)
		{
			Debug.Assert(neurons[0].Weghts.Length == prevNeuronsCount);

			double[] newGradSum = new double[prevNeuronsCount];
			for (int prevNeuronIndex = 0; prevNeuronIndex < newGradSum.Length; prevNeuronIndex++)
			{
				for (int neuronIndex = 0; neuronIndex < neurons.Length; neuronIndex++)
				{
					newGradSum[prevNeuronIndex] +=
						localGradients[neuronIndex] * neurons[neuronIndex].Weghts[prevNeuronIndex];
				}
			}
			return newGradSum;
		}
	}

	class FullyConnectedLayer : BaseLayer
	{
		protected double[] LastIutput;

		public FullyConnectedLayer(int inputsCount, int outputCount)
			: base(inputsCount, outputCount) { }

		public override double[] CalcOutput(double[] input)
		{
			LastIutput = input; // as layer has connections to all previous neurons.

			for (int i = 0; i < Neurons.Length; i++)
			{
				LastOutput[i] = Neurons[i].Activate(input);
			}
			return LastOutput;
		}

		public void BackwardPass(double[] gradSums, out double[] newGradSum)
        {
            // Local gradients for hidden neurons.
            double[] localGradients = new double[Neurons.Length];
            for (int i = 0; i < localGradients.Length; i++)
            {
                localGradients[i] = Neurons[i].DeactivateByValue(LastOutput[i]) * gradSums[i];
            }

			newGradSum = CalcGradSums(localGradients, Neurons, LastIutput.Length);
            CorrectWeights(Neurons, LastIutput, localGradients);
        }
	}

	class OutputLayer : FullyConnectedLayer
    {
        public OutputLayer(int inputsCount, int outputCount)
            : base(inputsCount, outputCount) { }

        public void BackwardPassOnOutput(
			double[] target, double[] networkOutput, out double[] newGradSum)
        {
			// Local gradients for output neurons.
			double[] localGradients = new double[Neurons.Length];
            for (int neuronIndex = 0; neuronIndex < Neurons.Length; neuronIndex++)
            {
                Neuron neuron = Neurons[neuronIndex];
				localGradients[neuronIndex] =
                    Config.E_ErroFunnDev(networkOutput, target, neuronIndex)
                    * (neuron.DeactivateByValue(networkOutput[neuronIndex]));
            }

			newGradSum = CalcGradSums(localGradients, Neurons, LastIutput.Length);
			CorrectWeights(Neurons, LastIutput, localGradients);
        }
    }

	class Network
	{
		private const string FileName = "data.xml";
		private readonly FullyConnectedLayer _hiddenLayer;
		private readonly OutputLayer _outputLayer;

		public Network() : this(2, 4, 2) { }

		public Network(FullyConnectedLayer hiddenLayer, OutputLayer outputLayer)
		{
			_hiddenLayer = hiddenLayer; _outputLayer = outputLayer;
		}

		public Network(int inputCount, int hiddenCount, int outputCount)
			: this(
                new FullyConnectedLayer(inputCount, hiddenCount),
                new OutputLayer(hiddenCount, outputCount)) { }

		public double[] ComputeOutput(double[] input)
		{
			// TODO: place cycle here
			double[] t1 = _hiddenLayer.CalcOutput(input);
			double[] t2 = _outputLayer.CalcOutput(t1);
			return t2;
		}

		public void BackwardPass(double[] networkOutput, double[] target)
		{
			double[] localGrads;
			double[] newLocalGrads;
			_outputLayer.BackwardPassOnOutput(target, networkOutput, out newLocalGrads);
			localGrads = newLocalGrads;
			_hiddenLayer.BackwardPass(localGrads, out newLocalGrads);
		}
    }

	class Neuron
	{
        public readonly double[] Weghts;

	    public Neuron(int inputWeghtsCount)
	    {
            Weghts = new double[inputWeghtsCount];
	        // Real problem was here!
	        for (int i = 0; i < Weghts.Length; i++)
	        {
                Weghts[i] = Helpers.GetSmallRandom();
	        }
	    }

        public Neuron(double[] inputWeghts)
		{
            Weghts = inputWeghts;
        }

        public double Activate(double[] input)
        {
            Debug.Assert(Weghts.Length == input.Length);

            double sum = 0;
        	for (int i = 0; i < input.Length; i++)
        	{
        		sum += input[i] * Weghts[i];
        	}
            return ActFun(sum);
        }

        private static double ActFun(double arg) => Math.Pow(1 + Math.Exp(-arg), -1);

        public double DeactivateByValue(double arg)
        {
        	// TODO:
        	// thereticaly f'(arg) = ActFun'[arg]
        	// but 
        	return arg * (1 - arg);
        }
    }
}
