using System;
using System.Collections.Generic;
using System.Diagnostics;
using MoreLinq;
using System.Windows;
using System.Numerics;

namespace NN
{
    public class Network
    {

        public List<Layer> layers = new List<Layer>();

        //first index determines layer, second determines ending node, third determines starting node
        Connection[][][] connections;
        public double learningRate = 1;

        public Network(List<Layer> layers1)
        {
            layers = layers1;
            connections = new Connection[layers.Count - 1][][];

            //initialize connections array
            for (int i = 0; i < layers.Count - 1; i++)
            {
                connections[i] = new Connection[layers[i + 1].nodes.Count][];
                for (int j = 0; j < layers[i + 1].nodes.Count; j++)
                {
                    connections[i][j] = new Connection[layers[i].nodes.Count];
                }
            }

            //make connections
            for (int i = 0; i < connections.Length; i++)
            {
                for (int j = 0; j < connections[i].Length; j++)
                {
                    for (int k = 0; k < connections[i][j].Length; k++)
                    {
                        connections[i][j][k] = new Connection(layers[i].nodes[k], layers[i + 1].nodes[j]);
                    }
                }
            }
        }

        public void PrintResults(Image image)
        {
            double loss = LossCalc(image);

            int guess = 0;
            Node highest = layers[layers.Count - 1].nodes[0];
            foreach(Node node in layers.Find(x => x.number == layers.Count - 1).nodes)
            {
                if(node.output > highest.output)
                {
                    highest = node;
                }

                guess = Array.IndexOf(layers[layers.Count - 1].nodes.ToArray(), highest);
            }

            int actual = image.Imagelabel;

            Console.WriteLine("Guess: " + guess);
            Console.WriteLine("Actual: " + actual);
            Console.WriteLine("Loss: " + loss);
        }

        public void SetInputs(Image image)
        {
            double[] pixels = new double[image.Imagepixels.Length * image.Imagepixels[0].Length];

            for (int k = 0; k < pixels.Length;)
            {
                for (int i = 0; i < image.Imagepixels.Length; i++)
                {
                    for (int j = 0; j < image.Imagepixels[i].Length; j++)
                    {
                        double value = image.Imagepixels[i][j];
                        pixels[k] = value / 255;
                        k++;
                    }
                }
            }

            int h = 0;
            foreach (Node node in layers.Find(x => x.number == 0).nodes)
            {
                node.input = pixels[h];
                h++;
            }
        }

        public void FeedForward()
        {
            for (int i = 0; i < connections.Length; i++)
            {
                for (int j = 0; j < connections[i].Length; j++)
                {
                    for (int k = 0; k < connections[i][j].Length; k++)
                    {
                        connections[i][j][k].start.Activate();
                        connections[i][j][k].FeedForward();
                    }
                }
            }

            foreach (Node node in layers[layers.Count - 1].nodes)
            {
                node.Activate();
            }
        }

        public int[] GetDesiredOutput(Image image)
        {
            int imlbl = image.Imagelabel;
            switch(imlbl)
            {
                case 0:
                    return new int[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                case 1:
                    return new int[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
                case 2:
                    return new int[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
                case 3:
                    return new int[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
                case 4:
                    return new int[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
                case 5:
                    return new int[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
                case 6:
                    return new int[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
                case 7:
                    return new int[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
                case 8:
                    return new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
                case 9:
                    return new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
                default:
                    Debug.WriteLine("This is a problem. The expected output for the network is somehow nothing.");
                    return new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            }
        }

        public void ErrorCalc(Image image)
        {
            int i = 0;
            int[] desiredOutput = GetDesiredOutput(image);
            foreach(Node node in layers.Find(x => x.number == layers.Count - 1).nodes)
            {
                node.error = desiredOutput[i] - node.output;
                i++;
            }
        }

        public double LossCalc(Image image)
        {
            int i = 0;
            double output = 0;
            int[] desiredOutput = GetDesiredOutput(image);
            foreach (Node node in layers.Find(x => x.number == layers.Count - 1).nodes)
            {
                output += Math.Pow(desiredOutput[i] - node.output, 2);
            }
            return output;
        }

        public void Backpropogate(Image image)
        {
            ErrorCalc(image);
            //node.error += (connection.weight / shareEndSum) * connection.end.error;

            //iterate through weights
            for (int i = connections.Length - 1; i >= 0; i--)
            {
                for (int j = 0; j < connections[i].Length; j++)
                {
                    for (int k = 0; k < connections[i][j].Length; k++)
                    {
                        Connection TopWeight = connections[i][j][k]; //a weight
                        Connection[] BottomWeights = Array.FindAll(connections[i][j], x => x.end == connections[i][j][k].end); //find all weights connected to the same ending
                        double BottomWeightsSum = 0;

                        //get sum
                        foreach(Connection connection in BottomWeights)
                        {
                            BottomWeightsSum += connection.weight;
                        }

                        TopWeight.start.error += (TopWeight.weight / BottomWeightsSum) * TopWeight.end.error;
                    }
                }
            }
        }

        public double[][][] WeightCalculations()
        {
            double[][][] output = new double[layers.Count - 1][][];
            int r = 0;
            foreach (Layer layer in layers.FindAll(x => x.number != 0))
            {

                double[] outputs = new double[layer.nodes.Count];
                double[] errors = new double[layer.nodes.Count];
                double[] gradients = new double[layer.nodes.Count];
                double[][] deltas = new double[gradients.Length][];

                //initialize deltas
                for (int i = 0; i < deltas.Length; i++)
                {
                    deltas[i] = new double[layers[layer.number - 1].nodes.Count];
                }

                for (int i = 0; i < layer.nodes.Count; i++)
                {
                    layer.nodes[i].output = layer.nodes[i].output * (1 - layer.nodes[i].output); //apply derivative of sigmoid to output
                    outputs[i] = layer.nodes[i].output;
                    errors[i] = layer.nodes[i].error;
                }


                //Element-wise multiplication of the arrays + learning rate (step size)
                for (int i = 0; i < gradients.Length; i++)
                {
                    gradients[i] = errors[i] * outputs[i];
                    gradients[i] *= learningRate;
                }

                //calculate deltas
                for (int i = 0; i < gradients.Length; i++)
                {
                    for (int j = 0; j < layers[layer.number - 1].nodes.Count; j++)
                    {
                        deltas[i][j] = gradients[i] * layers[layer.number - 1].nodes[j].output;
                    }
                }

                output[r] = deltas;
                r++;
            }
            return output;
        }

        public void UpdateWeights(double[][][] deltas)
        {
            for (int i = 0; i < connections.Length; i++)
            {
                for (int j = 0; j < connections[i].Length; j++)
                {
                    for (int k = 0; k < connections[i][j].Length; k++)
                    {
                        connections[i][j][k].weight += deltas[i][j][k];
                    }
                }
            }
        }

        public void ResetForNext()
        {
            foreach (Layer layer in layers)
            { 
                foreach (Node node in layer.nodes)
                {
                    node.input = 0;
                    node.output = 0;
                    node.error = 0;
                }
            }

            for (int i = 0; i < connections.Length; i++)
            {
                for (int j = 0; j < connections[i].Length; j++)
                {
                    for (int k = 0; k < connections[i][j].Length; k++)
                    {
                        connections[i][j][k].errorResponsibility = 0;
                    }
                }
            }
        }
    }
    public class Layer
    {
        public Layer(List<Node> nodes1, int number1)
        {
            nodes = nodes1;
            number = number1;
        }
        public List<Node> nodes = new List<Node>();
        public int number;
    }
    public class Connection
    {
        public Connection(Node beginning, Node ending)
        {
            start = beginning;
            end = ending;
            errorResponsibility = 0;
            weight = (Util.globalRandom.NextDouble() * (1 + 1)) - 1;
        }

        public Node start;
        public Node end;
        public double weight;
        public double errorResponsibility;

        public void FeedForward()
        {
            end.input += start.output * weight;
        }
    }
    public class Node
    {
        public Node(ActivationFunction function)
        {
            input = 1;
            output = 0;
            error = 0;
            activation = function;
        }

        public double input;
        public double output;
        public ActivationFunction activation;
        public double error;

        public void Activate()
        {
            if (activation == ActivationFunction.Identity)
            {
                output = input;
            }
            else if (activation == ActivationFunction.Binary)
            {
                if (input >= 0)
                {
                    output = 1;
                }
                else
                {
                    output = -1;
                }
            }
            else if (activation == ActivationFunction.HardLimit)
            {
                if (input >= 0)
                {
                    output = 1;
                }
                else
                {
                    output = 0;
                }
            }
            else if (activation == ActivationFunction.Sigmoid)
            {
                output = (double)1 / (1 + Math.Pow(Math.E, -input));
            }
            else if (activation == ActivationFunction.ReLU)
            {
                if (input >= 0)
                {
                    output = input;
                }
                else
                {
                    output = 0;
                }
            }
        }
    }
    public enum ActivationFunction
    {
        Identity,
        Binary,
        HardLimit,
        Sigmoid,
        ReLU,
    }
}

