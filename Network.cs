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
        List<Layer> layers = new List<Layer>();
        List<Connection> connections = new List<Connection>();
        public double learningRate = 0.2;

        public Network(List<Layer> layers1)
        {
            layers = layers1;
            foreach(Layer layer in layers1)
            {
                if (layer.number != layers1.Count - 1)
                {
                    foreach (Node node in layer.nodes)
                    {

                        foreach (Node node1 in layers1.Find(x => x.number == layer.number + 1).nodes)
                        {
                            connections.Add(new Connection(node, node1));
                        }

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
            foreach (Connection connection in connections)
            {
                connection.FeedForward();
            }

            foreach (Node node in layers.Find(x => x.number == layers.Count - 1).nodes)
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
            for(int i = layers.Count - 2; i > 0; i--)
            {
                foreach(Node node in layers[i].nodes)
                {
                    List<Connection> shareStart = connections.FindAll(x => x.start == node);
                    List<Connection> shareEnd = new List<Connection>();

                    foreach(Connection connection in shareStart)
                    {
                        shareEnd = connections.FindAll(x => x.end == connection.end);
                        double shareEndSum = 0;
                        foreach(Connection connection1 in shareEnd)
                        {
                            shareEndSum += connection1.weight;
                        }

                        node.error += (connection.weight / shareEndSum) * connection.end.error;
                    }
                }
            }
        }

        public void UpdateWeights()
        {
            foreach(Layer layer in layers)
            {
                if (layer.number != 0)
                {
                    double[] outputs = new double[layer.nodes.Count];
                    double[] errors = new double[layer.nodes.Count];
                    double[] gradients = new double[layer.nodes.Count];
                    double[][] deltas = new double[gradients.Length][];
                    double[] deltas1D = new double[gradients.Length * layers[layer.number - 1].nodes.Count];
                    
                    //initialize deltas
                    for(int i = 0; i < deltas.Length; i++)
                    {
                        deltas[i] = new double[layers[layer.number - 1].nodes.Count];
                    }

                    for (int i = 0; i < layer.nodes.Count; i++)
                    {
                        //node.output = node.output * (1 - node.output); //apply derivitave of sigmoid to output
                        //node.output
                        layer.nodes[i].output = layer.nodes[i].output * (1 - layer.nodes[i].output); //apply derivative of sigmoid to output
                        outputs[i] = layer.nodes[i].output;
                        errors[i] = layer.nodes[i].error;
                    }

                    //Vector<double> vector = new Vector<double>(outputs);

                    //Element-wise multiplication of the arrays + learning rate (step size)
                    for (int i = 0; i < outputs.Length; i++)
                    {
                        gradients[i] = errors[i] * outputs[i];
                        gradients[i] *= learningRate;
                    }

                    //calculate deltas
                    for (int i = 0; i < gradients.Length; i++)
                    {
                        for(int j = 0; j < layers[layer.number - 1].nodes.Count; j++)
                        {
                            deltas[i][j] = gradients[i] * layers[layer.number - 1].nodes[j].output;
                        }
                    }

                    //transform to 1d array
                    int k = 0;
                    for (int i = 0; i < deltas.Length; i++)
                    {
                        for (int j = 0; j < deltas[i].Length; j++)
                        {
                            deltas1D[k] = deltas[i][j];
                            k++;
                        }
                    }

                    k = 0;
                    foreach(Connection connection in connections.FindAll(x => layer.nodes.Contains(x.start)))
                    {
                        connection.weight += deltas1D[k];
                        k++;
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

            foreach(Connection connection in connections)
            {
                connection.errorResponsibility = 0;
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
        public Connection(Node node1, Node node2)
        {
            start = node1;
            end = node2;
            errorResponsibility = 0;
            weight = (Util.globalRandom.NextDouble() * (1 + 1)) - 1;
        }

        public Node start;
        public Node end;
        public double weight;
        public double errorResponsibility;

        public void FeedForward()
        {
            start.Activate();
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

