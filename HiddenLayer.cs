using System;

namespace MO_31_2_Shurpaev_Elizabeth.NeuroNet
{
    class HiddenLayer : Layer
    {
        public HiddenLayer(int non, int nopn, NeuronType nt, string nm_Layer) : base(non, nopn, nt, nm_Layer) { }

        public override void Recognize(Network net, Layer nextLayer)
        {
            double[] hidden_out = new double[numofneurons];
            Random rand = new Random();

            for (int i = 0; i < numofneurons; i++)
            {
                if (isTraining)
                {
                    // Dropout: отключаем часть нейронов
                    dropoutMask[i] = rand.NextDouble() > dropoutRate;
                    // Inverted Dropout: делим на (1-p)
                    hidden_out[i] = dropoutMask[i] ? (neurons[i].Output / (1.0 - dropoutRate)) : 0.0;
                }
                else
                {
                    hidden_out[i] = neurons[i].Output;
                }
            }
            nextLayer.Data = hidden_out;
        }

        public override double[] BackwardPass(double[] gr_sums)
        {
            double[] gr_sum = new double[numofprevneurons];

            for (int j = 0; j < numofprevneurons; j++)
            {
                double sum = 0;
                for (int k = 0; k < numofneurons; k++)
                {
                    // Градиент проходит только через активные нейроны
                    if (dropoutMask[k])
                    {
                        sum += neurons[k].Weights[j + 1] * neurons[k].Derivative * gr_sums[k];
                    }
                }
                gr_sum[j] = sum;
            }

            for (int i = 0; i < numofneurons; i++)
            {
                // Если нейрон был отключен Dropout-ом, веса не корректируем
                if (!dropoutMask[i] && isTraining) continue;

                for (int n = 0; n < numofprevneurons + 1; n++)
                {
                    double delwat;
                    double inputVal = (n == 0) ? 1.0 : neurons[i].Inputs[n - 1];
                    
                    delwat = momentum * lastdeltaweights[i, n] + learningrate * neurons[i].Derivative * gr_sums[i] * inputVal;

                    lastdeltaweights[i, n] = delwat;
                    neurons[i].Weights[n] += delwat;
                }
            }
            return gr_sum;
        }
    }
}