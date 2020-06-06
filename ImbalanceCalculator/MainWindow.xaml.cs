using System.Diagnostics;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace ImbalanceCalculator
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            sliders = new Slider[] { Slider0, Slider1, Slider2, Slider3, Slider4, Slider5, Slider6 };
            textBoxes = new TextBox[] { TextBox0, TextBox1, TextBox2, TextBox3, TextBox4, TextBox5, TextBox6 };
            pctLabels = new TextBlock[] { PctLabel0, PctLabel1, PctLabel2, PctLabel3, PctLabel4, PctLabel5, PctLabel6 };
        }

        private int total = 0;
        private double C2 = 0;
        private double IR = 0;
        private readonly Slider[] sliders;
        private readonly TextBox[] textBoxes;
        private readonly TextBlock[] pctLabels;

        private void UpdateStats()
        {
            // Update total
            total = sliders.Sum(s => (int)s.Value);
            TotalTextBlock.Text = $"Total = {total}";

            // Update pct
            if (total != 0)
            {
                for (int i = 0; i < 7; i++)
                {
                    pctLabels[i].Text = $"{sliders[i].Value / total * 100:F}%";
                }
            }
            else
            {
                for (int i = 0; i < 7; i++)
                {
                    pctLabels[i].Text = "?%";
                }
            }

            // Update IR
            int classCount = sliders.Count(s => s.Value > 0);
            double beforeSum = (classCount - 1) / (double)classCount;
            double sum = 0;
            foreach (int inputNumber in sliders.Select(s => (int)s.Value))
            {
                sum += (double)inputNumber / (total - inputNumber);
            }
            IR = beforeSum * sum;
            IrTextBlock.Text = $"IR = {IR:F4}";

            // Update C2
            C2 = 1.0 - 1.0 / IR;
            C2TextBlock.Text = $"C2 = {C2:F4}";
        }

        private void Slider0_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            TextBox0.Text = Slider0.Value.ToString();
            UpdateStats();
        }

        private void TextBox0_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (int.TryParse(TextBox0.Text, out int label0Input) && label0Input >= 0 && label0Input <= Slider0.Maximum)
            {
                TextBox0.Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 0));
                Slider0.Value = label0Input;
            }
            else
            {
                TextBox0.Foreground = new SolidColorBrush(Color.FromRgb(255, 0, 0));
            }
            UpdateStats();
        }

        private void Slider1_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            TextBox1.Text = Slider1.Value.ToString();
            UpdateStats();
        }

        private void TextBox1_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (int.TryParse(TextBox1.Text, out int label1Input) && label1Input >= 0 && label1Input <= Slider1.Maximum)
            {
                TextBox1.Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 0));
                Slider1.Value = label1Input;
            }
            else
            {
                TextBox1.Foreground = new SolidColorBrush(Color.FromRgb(255, 0, 0));
            }
            UpdateStats();
        }

        private void Slider2_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            TextBox2.Text = Slider2.Value.ToString();
            UpdateStats();
        }

        private void TextBox2_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (int.TryParse(TextBox2.Text, out int label2Input) && label2Input >= 0 && label2Input <= Slider2.Maximum)
            {
                TextBox2.Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 0));
                Slider2.Value = label2Input;
            }
            else
            {
                TextBox2.Foreground = new SolidColorBrush(Color.FromRgb(255, 0, 0));
            }
            UpdateStats();
        }


        private void Slider3_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            TextBox3.Text = Slider3.Value.ToString();
            UpdateStats();
        }

        private void TextBox3_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (int.TryParse(TextBox3.Text, out int label3Input) && label3Input >= 0 && label3Input <= Slider3.Maximum)
            {
                TextBox3.Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 0));
                Slider3.Value = label3Input;
            }
            else
            {
                TextBox3.Foreground = new SolidColorBrush(Color.FromRgb(255, 0, 0));
            }
            UpdateStats();
        }

        private void Slider4_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            TextBox4.Text = Slider4.Value.ToString();
            UpdateStats();
        }

        private void TextBox4_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (int.TryParse(TextBox4.Text, out int label4Input) && label4Input >= 0 && label4Input <= Slider4.Maximum)
            {
                TextBox4.Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 0));
                Slider4.Value = label4Input;
            }
            else
            {
                TextBox4.Foreground = new SolidColorBrush(Color.FromRgb(255, 0, 0));
            }
            UpdateStats();
        }

        private void Slider5_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            TextBox5.Text = Slider5.Value.ToString();
            UpdateStats();
        }

        private void TextBox5_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (int.TryParse(TextBox5.Text, out int label5Input) && label5Input >= 0 && label5Input <= Slider5.Maximum)
            {
                TextBox5.Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 0));
                Slider5.Value = label5Input;
            }
            else
            {
                TextBox5.Foreground = new SolidColorBrush(Color.FromRgb(255, 0, 0));
            }
            UpdateStats();
        }

        private void Slider6_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            TextBox6.Text = Slider6.Value.ToString();
            UpdateStats();
        }

        private void TextBox6_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (int.TryParse(TextBox6.Text, out int label6Input) && label6Input >= 0 && label6Input <= Slider6.Maximum)
            {
                TextBox6.Foreground = new SolidColorBrush(Color.FromRgb(0, 0, 0));
                Slider6.Value = label6Input;
            }
            else
            {
                TextBox6.Foreground = new SolidColorBrush(Color.FromRgb(255, 0, 0));
            }
            UpdateStats();
        }

        private void A270Button_Click(object sender, RoutedEventArgs e)
        {
            int[] config = new int[] { 0, 12, 44, 156, 3, 5, 16 };
            for (int i = 0; i < 7; i++)
            {
                sliders[i].Value = config[i];
                textBoxes[i].Text = config[i].ToString();
            }
        }

        private void B739Button_Click(object sender, RoutedEventArgs e)
        {
            int[] config = new int[] { 28, 173, 157, 166, 10, 104, 34 };
            for (int i = 0; i < 7; i++)
            {
                sliders[i].Value = config[i];
                textBoxes[i].Text = config[i].ToString();
            }
        }

        private void CitationHyperlink_RequestNavigate(object sender, System.Windows.Navigation.RequestNavigateEventArgs e)
        {
            Process.Start("explorer.exe", e.Uri.ToString());
        }
    }
}
