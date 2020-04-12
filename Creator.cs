using System;
using System.Collections.Generic;
using System.Linq;

namespace NN
{
    class Program
    {
        static void Main(string[] args)
        {
            
            List<Layer> tempLayers = new List<Layer>();
            List<Node> tempNodes = new List<Node>();
            Training training;
            

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
            training = new Training { net = yay };
            training.TrainingManager();
        }
    }
}
