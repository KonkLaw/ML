using System;

namespace NeuralNetwork
{
    static class Helpers
    {
        private static readonly Random Random = new Random(1);

        public static void WriteLine(this double[] array, string comment = null)
        {
            Console.Write(comment);
            Console.Write(": ");
            foreach (double item in array)
            {
                Console.Write(item);
                Console.Write(" ");
            }
            Console.WriteLine();
        }

        public static double Avg(this double[] array)
        {
            double sum = 0;
            foreach (double item in array)
            {
                sum += item;
            }
            return sum / array.Length;
        }

        public static double GetSmallRandom() => Random.NextDouble() / 100_000d;
    }
}