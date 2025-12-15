using System;
using System.Linq;

namespace MO_31_2_Shurpaev_Elizabeth.NeuroNet
{
    class Network
    {
        private InputLayer input_layer = null;
        private HiddenLayer hidden_layer1 = new HiddenLayer(71, 15, NeuronType.Hidden, nameof(hidden_layer1));
        private HiddenLayer hidden_layer2 = new HiddenLayer(33, 71, NeuronType.Hidden, nameof(hidden_layer2));
        private OutputLayer output_layer = new OutputLayer(10, 33, NeuronType.Output, nameof(output_layer));

        private double[] fact = new double[10];
        private double[] e_error_avr;
        private double[] accuracy_avr;
        public double[] Fact { get => fact; }

        public double[] E_error_avr { get => e_error_avr; set => e_error_avr = value; }
        public double[] Accuracy_avr { get => accuracy_avr; set => accuracy_avr = value; }

        public Network() { }

        public void ForwardPass(Network net, double[] netInput)
        {
            net.hidden_layer1.Data = netInput;
            net.hidden_layer1.Recognize(null, net.hidden_layer2);
            net.hidden_layer2.Recognize(null, net.output_layer);
            net.output_layer.Recognize(net, null);
        }

        public void Train(Network net)
        {
            net.input_layer = new InputLayer(NetworkMode.Train);
            int epoches = 12;
            int totalSamples = net.input_layer.Trainset.GetLength(0);

            e_error_avr = new double[epoches];
            accuracy_avr = new double[epoches];

            for (int k = 0; k < epoches; k++)
            {
                double totalError = 0;
                int correctPredictions = 0;

                net.input_layer.Shuffling_Array_Rows(net.input_layer.Trainset);

                for (int i = 0; i < totalSamples; i++)
                {
                    // Извлекаем входные признаки (15 значений)
                    double[] tmpTrain = new double[15];
                    for (int j = 0; j < tmpTrain.Length; j++)
                        tmpTrain[j] = net.input_layer.Trainset[i, j + 1];

                    // Прямой проход
                    ForwardPass(net, tmpTrain);

                    // Истинная метка
                    int trueLabel = (int)net.input_layer.Trainset[i, 0];

                    // Находим предсказанный класс (индекс максимального выхода)
                    int predictedLabel = Array.IndexOf(net.fact, net.fact.Max());

                    // Учёт точности
                    if (predictedLabel == trueLabel)
                        correctPredictions++;

                    // Расчёт ошибки (MSE-style)
                    double tmpSumError = 0;
                    double[] errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == trueLabel)
                            errors[x] = 1.0 - net.fact[x];
                        else
                            errors[x] = -net.fact[x];

                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    totalError += tmpSumError / errors.Length;

                    // Обратное распространение ошибки
                    double[] temp_gsums2 = net.output_layer.BackwardPass(errors);
                    double[] temp_gsums1 = net.hidden_layer2.BackwardPass(temp_gsums2);
                    net.hidden_layer1.BackwardPass(temp_gsums1);
                }

                // Средние значения за эпоху
                e_error_avr[k] = totalError / totalSamples;
                accuracy_avr[k] = (double)correctPredictions / totalSamples;
            }

            // Освобождение ресурсов
            net.input_layer = null;

            // Загрузка весов из памяти (если используется)
            net.hidden_layer1.WeightInitialize(MemoryMode.SET, nameof(hidden_layer1) + "_memory.csv");
            net.hidden_layer2.WeightInitialize(MemoryMode.SET, nameof(hidden_layer2) + "_memory.csv");
            net.output_layer.WeightInitialize(MemoryMode.SET, nameof(output_layer) + "_memory.csv");
        }

        public void Test(Network net)
        {
            net.input_layer = new InputLayer(NetworkMode.Test);
            int epoches = 5;
            int totalSamples = net.input_layer.Testset.GetLength(0);

            e_error_avr = new double[epoches];
            accuracy_avr = new double[epoches];

            for (int k = 0; k < epoches; k++)
            {
                double totalError = 0;
                int correctPredictions = 0;

                net.input_layer.Shuffling_Array_Rows(net.input_layer.Testset);

                for (int i = 0; i < totalSamples; i++)
                {
                    double[] tmpTest = new double[15];
                    for (int j = 0; j < tmpTest.Length; j++)
                        tmpTest[j] = net.input_layer.Testset[i, j + 1];

                    ForwardPass(net, tmpTest);

                    int trueLabel = (int)net.input_layer.Testset[i, 0];
                    int predictedLabel = Array.IndexOf(net.fact, net.fact.Max());

                    if (predictedLabel == trueLabel)
                        correctPredictions++;

                    double tmpSumError = 0;
                    double[] errors = new double[net.fact.Length];
                    for (int x = 0; x < errors.Length; x++)
                    {
                        if (x == trueLabel)
                            errors[x] = 1.0 - net.fact[x];
                        else
                            errors[x] = -net.fact[x];

                        tmpSumError += errors[x] * errors[x] / 2;
                    }
                    totalError += tmpSumError / errors.Length;
                }

                e_error_avr[k] = totalError / totalSamples;
                accuracy_avr[k] = (double)correctPredictions / totalSamples;
            }
        }
    }
}