using System;
using System.IO;

namespace MO_31_2_Shurpaev_Elizabeth.NeiroNet
{
    abstract class Layer
    {
        //Поля
        protected string name_Layer; //название слоя
        string pathDirWeights; //путь к каталогу, где находится файл синаптических весов
        string pathFileWeights; //путь к файлу саниптическов весов
        protected int numofneurons; //число нейронов текущего слоя
        protected int numofprevneurons; //число нейронов предыдущего слоя
        protected const double learningrate = 0.0035; //скорость обучения
        protected const double momentum = 0.9; //момент инерции
        protected double[,] lastdeltaweights; //веса предыдущей итерации
        protected Neuron[] neurons; //массив нейронов текущего слоя
        //Свойства
        public Neuron[] Neurons { get => neurons; set => neurons = value; }
        public double[] Data //Передача входных сигналов на нейроны слоя и авктиватор
        {
            set
            {
                for (int i = 0; i < numofneurons; i++)
                {
                    Neurons[i].Activator(value);
                }
            }
        }

        //Конструктор
        protected Layer(int non, int nopn, NeuronType nt, string nm_Layer)
        {
            numofneurons = non; //количество нейронов текущего слоя
            numofprevneurons = nopn; //количество нейронов предыдущего слоя
            Neurons = new Neuron[non]; //определение массива нейронов
            name_Layer = nm_Layer; //наиминование слоя
            pathDirWeights = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            pathFileWeights = pathDirWeights + name_Layer + "_memory.csv";

            double[,] Weights; //временный массив синаптических весов
            lastdeltaweights = new double[non, nopn + 1];

            if (File.Exists(pathFileWeights)) //определяет существует ли pathFileWeights
                Weights = WeightInitialize(MemoryMode.GET, pathFileWeights); //считывает данные из файла
            else
            {
                Directory.CreateDirectory(pathDirWeights);
                Weights = WeightInitialize(MemoryMode.INIT, pathFileWeights);
            }

            for (int i = 0; i < non; i++) //цикл формирования нейронов слоя и заполнения
            {
                double[] tmp_weights = new double[nopn + 1];
                for (int j = 0; j < nopn; j++)
                {
                    tmp_weights[j] = Weights[i, j];
                }
                Neurons[i] = new Neuron(tmp_weights, nt); //заполнение массива нейронами
            }
        }

        //Метод работы с массивом синаптических весов слоя
        public double[,] WeightInitialize(MemoryMode mm, string path)
        {
            char[] delim = new char[] { ';', ' ' };
            string[] tmpStrWeights;
            double[,] weights = new double[numofneurons, numofprevneurons + 1];

            switch (mm)
            {
                // парсинг в тип double строкового формата веса нейронов из csv - получает значения весов нейронов
                case MemoryMode.GET:
                    tmpStrWeights = File.ReadAllLines(path);        // считывание строк текстового файла csv весов нейрона (в tmpStrWeights каждый i-ый элемент это строка весов)
                    string[] memory_elemnt; // массив, где каждый i-ый элемент это один вес нейрона (берётся одна строка из tmpStrWeights)

                    // строка весов нейронов
                    for (int i = 0; i < numofneurons; i++)
                    {
                        memory_elemnt = tmpStrWeights[i].Split(delim);  // разбивает строку
                        // каждый отдельный вес нейрона
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            weights[i, j] = double.Parse(memory_elemnt[j].Replace(',', '.'),
                                System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }
                    break;

                // парсинг в строковой формат веса нейронов в csv (обратный MemoryMode.GET) - сохраняет готовые веса нейронов
                case MemoryMode.SET:
                    tmpStrWeights = new string[numofneurons]; // создаём строку из весов нейрона (tmpStrWeights это массив, где каждый i-ый элемент это строка весов) 
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

                // инициализация весов для нейронов
                case MemoryMode.INIT:
                    // Защита от деления на ноль
                    if (numofprevneurons <= 0)
                        throw new InvalidOperationException("Количество нейронов предыдущего слоя должно быть положительным при инициализации весов.");

                    Random random = new Random(); // или используйте внеклассовый экземпляр Random для лучшей случайности

                    // Параметр активации (Leaky ReLU)
                    const double alpha = 0.01;

                    // Fan-in — количество входов (обычно не включает bias)
                    int fanIn = numofprevneurons;

                    // Граница для равномерного распределения: Xavier/He для Leaky ReLU
                    double scale = Math.Sqrt(6.0 / ((1.0 + alpha * alpha) * fanIn));

                    // Инициализация весов
                    for (int i = 0; i < numofneurons; i++)
                    {
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            if (j == numofprevneurons)
                            {
                                // Bias (последний вес) — инициализируем нулём
                                weights[i, j] = 0.0;
                            }
                            else
                            {
                                // Веса: равномерное распределение в [-scale, +scale]
                                weights[i, j] = (random.NextDouble() * 2.0 - 1.0) * scale;
                            }
                        }
                    }

                    // Сохранение в CSV
                    string[] lines = new string[numofneurons];
                    for (int i = 0; i < numofneurons; i++)
                    {
                        string[] values = new string[numofprevneurons + 1];
                        for (int j = 0; j < numofprevneurons + 1; j++)
                        {
                            values[j] = weights[i, j]
                                .ToString(System.Globalization.CultureInfo.InvariantCulture)
                                .Replace('.', ',');
                        }
                        lines[i] = string.Join(";", values);
                    }
                    File.WriteAllLines(path, lines);
                    break;
            }
            return weights;
        }


        abstract public void Recognize(Network net, Layer nextLayer);
        abstract public double[] BackwardPass(double[] staff);
    }
}