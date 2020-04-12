using System;
using System.Collections.Generic;
using System.Linq;

namespace NN
{
    class Program
    {
        static void Main(string[] args)
        {
            long EPOCHS = 10000000000000;
            int iterationsPerEpoch = 10;
            int iterationsDoneInEpoch = 0;
            bool testing = false;

            double[][][] bob;
            double[][][] beep = null;
            List<Layer> tempLayers = new List<Layer>();
            List<Node> tempNodes = new List<Node>();
            Decoder decoder = new Decoder();
            List<Image> images = decoder.Decode(testing);
            List<double[][][]> deltasOverEPOCH = new List<double[][][]>();
            List<double> losses = new List<double>();

            for (int i = 0; i < 784; i++)
            {
                tempNodes.Add(new Node(ActivationFunction.Identity));
            }
            tempLayers.Add(new Layer(tempNodes, 0));
            tempNodes = new List<Node>();

            for (int i = 0; i < 32; i++)
            {
                tempNodes.Add(new Node(ActivationFunction.Sigmoid));
            }
            tempLayers.Add(new Layer(tempNodes, 1));
            tempNodes = new List<Node>();

            for (int i = 0; i < 16; i++)
            {
                tempNodes.Add(new Node(ActivationFunction.Sigmoid));
            }
            tempLayers.Add(new Layer(tempNodes, 2));
            tempNodes = new List<Node>();

            for (int i = 0; i < 10; i++)
            {
                tempNodes.Add(new Node(ActivationFunction.Sigmoid));
            }
            tempLayers.Add(new Layer(tempNodes, 3));
            tempNodes = new List<Node>();
            Network yay = new Network(tempLayers);

            int imagebeep = 0;
            for(int i = 0; i < EPOCHS; i++)
            {
                for (iterationsDoneInEpoch = 0; iterationsDoneInEpoch < iterationsPerEpoch; iterationsDoneInEpoch++)
                {
                    yay.SetInputs(images[imagebeep]);
                    yay.FeedForward();
                    losses.Add(yay.LossCalc(images[imagebeep]));
                    yay.PrintResults(images[imagebeep]);
                    yay.Backpropogate(images[imagebeep]);
                    beep = yay.WeightCalculations();
                    deltasOverEPOCH.Add(beep);
                    yay.ResetForNext();
                    imagebeep++;
                }
                //imagebeep = 0;
                Console.WriteLine("EPOCH " + i + " done");
                //Average iterations
                bob = new double[beep.Length][][];
                beep.CopyTo(bob, 0);

                for (int h = 0; h < bob.Length; h++)
                {
                    for (int j = 0; j < bob[h].Length; j++)
                    {
                        for (int k = 0; k < bob[h][j].Length; k++)
                        {
                            double tempSum = 0;
                            for(int z = 0; z < deltasOverEPOCH.Count; z++)
                            {
                                tempSum += deltasOverEPOCH[z][h][j][k];
                            }
                            bob[h][j][k] = tempSum / deltasOverEPOCH.Count;
                        }
                    }
                }

                Console.WriteLine("Average loss over EPOCH: " + losses.Average());
                //System.Threading.Thread.Sleep(5000);
                yay.UpdateWeights(bob);
                losses.Clear();
                deltasOverEPOCH.Clear();
            }
        }
    }
}
