using System;
using System.Collections.Generic;

namespace NN
{
    class Program
    {
        static void Main(string[] args)
        {
            long iterations = 10000000000000;
            bool testing = false;

            List<Layer> tempLayers = new List<Layer>();
            List<Node> tempNodes = new List<Node>();
            Decoder decoder = new Decoder();
            List<Image> images = decoder.Decode(testing);

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
            
            for(int i = 0; i < iterations; i++)
            {
                yay.SetInputs(images[i]);
                yay.FeedForward();
                yay.PrintResults(images[i]);
                yay.Backpropogate(images[i]);
                yay.UpdateWeights();
                yay.ResetForNext();
            }
        }
    }
}
