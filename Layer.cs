using System;
using System.IO;

namespace MO_31_2_Shurpaev_Elizabeth.NeuroNet
{
    abstract class Layer
    {
        protected string name_Layer;
        string pathDirWeights;
        string pathFileWeights;
        protected int numofneurons;
        protected int numofprevneurons;
        protected const double learningrate = 0.04;
        protected const double momentum = 0.1;
        protected double[,] lastdeltaweights;
        protected Neuron[] neurons;

        // Поля для Dropout
        protected bool isTraining = false; 
        protected double dropoutRate = 0.2; // 20% нейронов будут отключаться
        protected bool[] dropoutMask; 

        public Neuron[] Neurons { get => neurons; set => neurons = value; }
        public bool IsTraining { get => isTraining; set => isTraining = value; }

        public double[] Data 
        {
            set
            {
                for (int i = 0; i < numofneurons; i++)
                {
                    Neurons[i].Activator(value);
                }
            }
        }

        protected Layer(int non, int nopn, NeuronType nt, string nm_Layer)
        {
            numofneurons = non;
            numofprevneurons = nopn;
            Neurons = new Neuron[non];
            name_Layer = nm_Layer;
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv";
            
            dropoutMask = new bool[non]; // Инициализация маски
            for (int i = 0; i < non; i++) dropoutMask[i] = true;

            double[,] Weights;
            lastdeltaweights = new double[non, nopn + 1];

            if (File.Exists(pathFileWeights))
                Weights = WeightInitialize(MemoryMode.GET, pathFileWeights);
            else
            {
                Directory.CreateDirectory(pathDirWeights);
                Weights = WeightInitialize(MemoryMode.INIT, pathFileWeights);
            }

            for (int i = 0; i < non; i++)
            {
                double[] tmp_weights = new double[nopn + 1];
                for (int j = 0; j <= nopn; j++) // Исправлено: включен порог
                {
                    tmp_weights[j] = Weights[i, j];
                }
                Neurons[i] = new Neuron(tmp_weights, nt);
            }
        }

        public double[,] WeightInitialize(MemoryMode mm, string path)
        {
            char[] delim = new char[] { ';', ' ' };
            string[] tmpStrWeights;
            double[,] weights = new double[numofneurons, numofprevneurons + 1];

            switch (mm)
            {
                case MemoryMode.GET:
                    tmpStrWeights = File.ReadAllLines(path);
                    for (int i = 0; i < numofneurons; i++)
                    {
                        string[] memory_elemnt = tmpStrWeights[i].Split(delim);
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = double.Parse(memory_elemnt[j].Replace(',', '.'),
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                case MemoryMode.SET:
                    tmpStrWeights = new string[numofneurons];
                    for (int i = 0; i < numofneurons; i++)
                    {
                        string[] memory_elemnt2 = new string[numofprevneurons + 1];
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            memory_elemnt2[j] = neurons[i].Weights[j]
                                .ToString(System.Globalization.CultureInfo.InvariantCulture)
                                .Replace('.', ',');
                        }
                        tmpStrWeights[i] = string.Join(";", memory_elemnt2);
                    }
                    File.WriteAllLines(path, tmpStrWeights);
                    break;

                case MemoryMode.INIT:
                    Random random = new Random();
                    double alpha = 0.01; 
                    int fanIn = numofprevneurons;
                    double limit = Math.Sqrt(6.0 / ((1.0 + alpha * alpha) * fanIn));

                    for (int i = 0; i < numofneurons; i++)
                    {
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            if (j == numofprevneurons) weights[i, j] = 0.0;
                            else weights[i, j] = (random.NextDouble() * 2.0 - 1.0) * limit;
                        }
                    }
                    break;
            }
            return weights;
        }

        abstract public void Recognize(Network net, Layer nextLayer);
        abstract public double[] BackwardPass(double[] staff);
    }
}