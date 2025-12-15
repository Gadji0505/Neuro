using System;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System.IO;
using MO_31_2_Shurpaev_Elizabeth.NeuroNet;

namespace MO_31_2_Shurpaev_Elizabeth
{
    public partial class FormMain : Form
    {
        private double[] inputPixels; // хранение состояния пикселей (0 - белый, 1 - чёрный)
        private Network network;

        //Конструктор
        public FormMain()
        {
            InitializeComponent();
            inputPixels = new double[15];
            network = new Network();
        }

        //Обработчик кнопки
        private void change_btn_onClick(object sender, EventArgs e)
        {
            if (((Button)sender).BackColor == Color.White)      // если белый
            {
                ((Button)sender).BackColor = Color.Black;       // то меняем на чёрный
                inputPixels[((Button)sender).TabIndex] = 1d;    // флаг состояния
            }
            else // если чёрный
            {
                ((Button)sender).BackColor = Color.White;       // то меняем на белый
                inputPixels[((Button)sender).TabIndex] = 0d;    // флаг состояния
            }
        }

        // Кнопка для добавления значения пикселей в тхт файл
        private void button_SaveTrainSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "train.txt";  // путь сохранения
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();     // значение счётчика numeric в form1 сохраняем

            // цикл добавления цифры и пикселей в строку для тхт файла
            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";     // отступ
            File.AppendAllText(path, tmpStr); // запись в файл
        }

        // Кнопка для проверки значений пикселей в тхт файл (повторяем прошлый обработчик, но меняем название файла)
        private void button_SaveTestSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "test.txt";
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
            {
                tmpStr += " " + inputPixels[i].ToString();
            }
            tmpStr += "\n";
            File.AppendAllText(path, tmpStr);
        }

        private void buttonRecognize_Click(object sender, EventArgs e)
        {
            network.ForwardPass(network, inputPixels);
            // Используем новые имена переменных
            label_Output.Text = network.Fact.ToList().IndexOf(network.Fact.Max()).ToString();
            label_Probability.Text = (100 * network.Fact.Max()).ToString("0.00") + " %";
        }

        private void buttonTrain_Click(object sender, EventArgs e)
        {
            network.Train(network);

            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_error_avr[i]);
                chart1.Series[0].Points.AddY(network.Accuracy_avr[i]);
            }

            MessageBox.Show("Обучение успешно завершено.", "Информация", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void buttonTest_Click(object sender, EventArgs e)
        {
            network.Test(network);

            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_error_avr[i]);
                chart1.Series[0].Points.AddY(network.Accuracy_avr[i]);
            }

            MessageBox.Show("Тестирование успешно завершено.", "Информация", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        // Новый обработчик для кнопки проверки файла весов
        private void button_CheckWeightsFile_Click(object sender, EventArgs e)
        {
            string weightsPath = AppDomain.CurrentDomain.BaseDirectory + "weights.txt";

            if (File.Exists(weightsPath))
            {
                try
                {
                    FileInfo fileInfo = new FileInfo(weightsPath);
                    if (fileInfo.Length > 0)
                    {
                        MessageBox.Show($"Файл весов существует.\nРазмер: {fileInfo.Length} байт\nПуть: {weightsPath}",
                            "Информация о файле весов", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    else
                    {
                        MessageBox.Show("Файл весов существует, но пустой.",
                            "Информация о файле весов", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Ошибка при проверке файла весов:\n{ex.Message}",
                        "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
            else
            {
                MessageBox.Show("Файл весов не найден.\nСоздайте его после обучения нейросети.",
                    "Файл не найден", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }

        // Обработчик загрузки формы (если нужен)
        private void FormMain_Load(object sender, EventArgs e)
        {
            // Инициализация при загрузке формы (если требуется)
        }
    }
}