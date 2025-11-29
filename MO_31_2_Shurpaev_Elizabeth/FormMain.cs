using System;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System.IO;
using MO_31_2_Shurpaev_Elizabeth.NeiroNet;

namespace MO_31_2_Shurpaev_Elizabeth
{
    public partial class FormMain : Form
    {
        private double[] inputPixels; // массив входных данных
        private Network network; //объявление нейросети

        //конструктор
        public FormMain()
        {
            InitializeComponent();
            inputPixels = new double[15];
            network = new Network();
        }

        private void Changing_State_Pixel_Button_Click(object sender, EventArgs e)
        {
            Button btn = (Button)sender;

            if (btn.BackColor == Color.White)
            {
                btn.BackColor = Color.Black;
                inputPixels[btn.TabIndex] = 1d;
            }
            else
            {
                btn.BackColor = Color.White;
                inputPixels[btn.TabIndex] = 0d;
            }
        }

        private void button_SaveTrainSample_Click(object sender, EventArgs e)
        {
            string path = AppDomain.CurrentDomain.BaseDirectory + "train.txt";
            string tmpStr = numericUpDown_NecessaryOutput.Value.ToString();

            for (int i = 0; i < inputPixels.Length; i++)
                tmpStr += " " + inputPixels[i].ToString();

            tmpStr += "\n";
            File.AppendAllText(path, tmpStr);
        }

        // Создание папки memory и файлов весов, если их нет
        private void FormMain_Load(object sender, EventArgs e)
        {
            try
            {
                string pathDir = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
                if (!Directory.Exists(pathDir))
                    Directory.CreateDirectory(pathDir);

                // создаём пустые файлы, если их нет
                string[] layerNames = { "InputLayer", "HiddenLayer", "OutputLayer" };
                foreach (var name in layerNames)
                {
                    string path = pathDir + name + "_memory.csv";
                    if (!File.Exists(path))
                    {
                        File.WriteAllText(path, ""); // создаём пустой
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Ошибка инициализации памяти: " + ex.Message,
                    "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void button_CheckWeightsFile_Click(object sender, EventArgs e)
        {
            string pathDir = AppDomain.CurrentDomain.BaseDirectory + "memory\\";
            if (!Directory.Exists(pathDir))
            {
                MessageBox.Show("Папка memory не найдена!", "Проверка",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            string[] files = Directory.GetFiles(pathDir, "*_memory.csv", SearchOption.TopDirectoryOnly);

            if (files.Length == 0)
            {
                MessageBox.Show("Файлы с весами не найдены!", "Проверка",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            else
            {
                string fileList = string.Join("\n", files.Select(f => Path.GetFileName(f)));
                MessageBox.Show($"Найдены файлы весов:\n{fileList}",
                    "Проверка", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        private void Button_Recognize_Click(object sender, EventArgs e)
        {
            network.ForwardPass(network, inputPixels);
            label_Output.Text = network.Fact.ToList().IndexOf(network.Fact.Max()).ToString();
            label_Probability.Text = (100 * network.Fact.Max()).ToString("0.00") + "%";
        }

        private void button_Training_Click(object sender, EventArgs e)
        {
            network.Train(network);

            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_error_avr[i]);
            }

            MessageBox.Show("Обучение успешно завершено.", "Информация",
                MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void buttonTest_Click(object sender, EventArgs e)
        {
            network.Test(network);

            for (int i = 0; i < network.E_error_avr.Length; i++)
            {
                chart_Eavr.Series[0].Points.AddY(network.E_error_avr[i]);
            }

            MessageBox.Show("Тестирование успешно завершено.", "Информация", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }
    }
}
