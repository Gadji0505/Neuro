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
            // Активируем режим обучения для Dropout
            net.hidden_layer1.IsTraining = true;
            net.hidden_layer2.IsTraining = true;

            int epoches = 50; // Увеличено, так как Dropout замедляет сходимость, но улучшает результат
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
                    double[] tmpTrain = new double[15];
                    for (int j = 0; j < tmpTrain.Length; j++)
                        tmpTrain[j] = net.input_layer.Trainset[i, j + 1];

                    ForwardPass(net, tmpTrain);

                    int trueLabel = (int)net.input_layer.Trainset[i, 0];
                    int predictedLabel = Array.IndexOf(net.fact, net.fact.Max());

                    if (predictedLabel == trueLabel) correctPredictions++;

                    double[] errors = new double[10];
                    double tmpSumError = 0;
                    for (int x = 0; x < 10; x++)
                    {
                        errors[x] = (x == trueLabel) ? (1.0 - net.fact[x]) : -net.fact[x];
                        tmpSumError += errors[x] * errors[x] / 2.0;
                    }
                    totalError += tmpSumError;

                    double[] temp_gsums2 = net.output_layer.BackwardPass(errors);
                    double[] temp_gsums1 = net.hidden_layer2.BackwardPass(temp_gsums2);
                    net.hidden_layer1.BackwardPass(temp_gsums1);
                }
                e_error_avr[k] = totalError / totalSamples;
                accuracy_avr[k] = (double)correctPredictions / totalSamples;
            }

            // Отключаем обучение
            net.hidden_layer1.IsTraining = false;
            net.hidden_layer2.IsTraining = false;

            net.hidden_layer1.WeightInitialize(MemoryMode.SET, "hidden_layer1_memory.csv");
            net.hidden_layer2.WeightInitialize(MemoryMode.SET, "hidden_layer2_memory.csv");
            net.output_layer.WeightInitialize(MemoryMode.SET, "output_layer_memory.csv");
        }

        public void Test(Network net)
        {
            net.input_layer = new InputLayer(NetworkMode.Test);
            net.hidden_layer1.IsTraining = false; // Dropout отключен
            net.hidden_layer2.IsTraining = false;

            int epoches = 1;
            int totalSamples = net.input_layer.Testset.GetLength(0);
            e_error_avr = new double[epoches];
            accuracy_avr = new double[epoches];

            double totalError = 0;
            int correctPredictions = 0;

            for (int i = 0; i < totalSamples; i++)
            {
                double[] tmpTest = new double[15];
                for (int j = 0; j < tmpTest.Length; j++)
                    tmpTest[j] = net.input_layer.Testset[i, j + 1];

                ForwardPass(net, tmpTest);

                int trueLabel = (int)net.input_layer.Testset[i, 0];
                if (Array.IndexOf(net.fact, net.fact.Max()) == trueLabel) correctPredictions++;
            }
            accuracy_avr[0] = (double)correctPredictions / totalSamples;
        }
    }
}