using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;

namespace NN
{
    public class Training
    {
        public Network net;
        List<Network> nets;
        long batches = 10000000000000;
        int batchSize = 10;
        int iterationsDoneInBatch = 0;
        int imageBeep = 0;
        int imageLabelTracker = 0;
        static bool testing = false;
        double[][][] bob;
        double[][][] beep = null;
        static Decoder decoder = new Decoder();
        List<Image> images = decoder.Decode(testing);
        List<double[][][]> deltasOverBatch = new List<double[][][]>();
        List<double> losses = new List<double>();

        public void TrainingManager()
        {
            for(int i = 0; i < batches; i++)
            {
                nets = DeepCopy(net, batchSize);
                Parallel.ForEach(nets, (network) =>
                {
                    if (imageLabelTracker > 9)
                    {
                        imageLabelTracker = 0;
                    }
                    Train(images.FindAll(x => x.Imagelabel == imageLabelTracker)[imageBeep], network);
                    imageLabelTracker++;
                    imageBeep++;
                });
                Update(net);
                Test(net, images[imageBeep + 1]);
            }
        }
        
        public void Train(Image image, Network network)
        {
            network.SetInputs(image);
            network.FeedForward();
            losses.Add(network.LossCalc(image));
            //network.PrintResults(image);
            network.Backpropogate(image);
            beep = network.WeightCalculations();
            deltasOverBatch.Add(beep);
            network.ResetForNext();
        }

        public void Update(Network network)
        {
            bob = new double[beep.Length][][];
            beep.CopyTo(bob, 0);

            for (int h = 0; h < bob.Length; h++)
            {
                for (int j = 0; j < bob[h].Length; j++)
                {
                    for (int k = 0; k < bob[h][j].Length; k++)
                    {
                        double tempSum = 0;
                        for (int z = 0; z < deltasOverBatch.Count; z++)
                        {
                            tempSum += deltasOverBatch[z][h][j][k];
                        }
                        bob[h][j][k] = tempSum;
                    }
                }
            }

            Console.WriteLine("Average loss over EPOCH: " + losses.Average());
            //System.Threading.Thread.Sleep(5000);
            network.UpdateWeights(bob);
            losses.Clear();
            deltasOverBatch.Clear();
        }

        public List<Network> DeepCopy(Network network, int count)
        {
            List<Network> output = new List<Network>();
            output.Add(network);

            for(int i = 0; i < count; i++)
            {
                output.Add(new Network(network.layers));
            }
            return output;
        }

        public void Test(Network network, Image image)
        {
            network.SetInputs(image);
            network.FeedForward();
            network.PrintResults(image);
        }
    }
}
