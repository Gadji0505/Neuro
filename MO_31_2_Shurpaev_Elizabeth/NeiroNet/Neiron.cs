using static System.Math;

namespace MO_31_2_Shurpaev_Elizabeth.NeiroNet
{
    class Neuron
    {
        //поля
        private NeuronType type; //тип нейрона
        private double[] weights; //его вес
        private double[] inputs; //его входы
        private double output; //выход
        private double derivative; //производная

        private double a = 0.01d;
        //Константы для функции активации

        //Свойства
        public double[] Weights { get => weights; set => weights = value; }
        public double[] Inputs { get => inputs; set => inputs = value; }
        public double Output { get => output; }
        public double Derivative { get => derivative; }
        //Конструктор
        public Neuron(double[] memoryWeights, NeuronType typeNeuron)
        {
            type = typeNeuron;
            weights = memoryWeights;
        }
        public void Activator(double[] i)
        {
            inputs = i; //передача вектора входного сигнала в массив входных данных нейрона
            double sum = weights[0];

            for (int j = 0; j < inputs.Length; j++)
            {
                sum += inputs[j] * weights[j + 1];
            }
            switch (type)
            {
                case NeuronType.Hidden:
                    output = LeakyReLU(sum);
                    derivative = LeakyReLU_Derivativator(sum);
                    break;
                case NeuronType.Output:
                    output = Exp(sum);
                    break;
            }
        }

        // функция активации нейрона
        private double LeakyReLU(double sum)
        {
            output = (sum >= 0) ? sum : a * sum;
            return output;
        }

        private double LeakyReLU_Derivativator(double sum)
        {
            derivative = (sum >= 0) ? 1.0 : a;
            return derivative;
        }

        //архитектура нейронов 15-71-31-10
    }
}