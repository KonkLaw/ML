namespace NeuralNetwork.Storing
{
    class StoringHelper
    {
        private class SerializedNetwork
        {
            public double[][] HiddenLayerWeights { get; set; }

            public double[][] OutputLayerWeights { get; set; }
        }

        //public string GetSerializedData(Network network)
        //{
        //
        //}
    }
}
