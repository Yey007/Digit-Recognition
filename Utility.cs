using System;
using System.IO;
using System.Collections.Generic;
using System.Buffers.Binary;

namespace NN
{
    public class Decoder
    {
        public List<Image> Decode(bool testing)
        {
            if (testing)
            {
                using (FileStream LabelsFile = new FileStream(@"C:\Users\utku_\source\repos\nn_hello_world\Test Data\t10k-labels-idx1-ubyte", FileMode.Open))
                {
                    using (FileStream ImagesFile = new FileStream(@"C:\Users\utku_\source\repos\nn_hello_world\Test Data\t10k-images-idx3-ubyte", FileMode.Open))
                    {
                        using (BinaryReader brImages = new BinaryReader(ImagesFile))
                        {
                            using (BinaryReader brLabels = new BinaryReader(LabelsFile))
                            {
                                List<Image> Images = new List<Image>();

                                brImages.BaseStream.Position = 0;
                                brLabels.BaseStream.Position = 0;

                                int magic1 = brImages.ReadInt32(); // discard
                                int numImages = brImages.ReadInt32();
                                int numRows = brImages.ReadInt32();
                                int numCols = brImages.ReadInt32();

                                int magic2 = brLabels.ReadInt32();
                                int numLabels = brLabels.ReadInt32();

                                byte[][] pixels = new byte[28][];
                                for (int i = 0; i < pixels.Length; ++i)
                                {
                                    pixels[i] = new byte[28];
                                }

                                byte b;

                                // each image
                                for (int di = 0; di < 60000; di++)
                                {
                                    for (int i = 0; i < 28; i++)
                                    {
                                        for (int j = 0; j < 28; j++)
                                        {

                                            b = brImages.ReadByte();
                                            pixels[i][j] = b;
                                        }
                                    }
                                    byte label = brLabels.ReadByte();
                                    //Images.Add(new Image { Imagepixels = pixels, Imagelabel = label, RowCount = numRows, ColCount = numCols });
                                }

                                return Images;
                            }
                        }
                    }
                }
            }
            else
            {
                using (FileStream LabelsFile = new FileStream(@"C:\Users\utku_\source\repos\nn_hello_world\Training Data\train-labels-idx1-ubyte", FileMode.Open))
                {
                    using (FileStream ImagesFile = new FileStream(@"C:\Users\utku_\source\repos\nn_hello_world\Training Data\train-images-idx3-ubyte", FileMode.Open))
                    {
                        using (BinaryReader brImages = new BinaryReader(ImagesFile))
                        {
                            using (BinaryReader brLabels = new BinaryReader(LabelsFile))
                            {
                                List<Image> Images = new List<Image>();

                                brImages.BaseStream.Position = 0;
                                brLabels.BaseStream.Position = 0;

                                int magic1 = brImages.ReadInt32(); // discard
                                int numImages = brImages.ReadInt32();
                                int numRows = brImages.ReadInt32();
                                int numCols = brImages.ReadInt32();

                                int magic2 = brLabels.ReadInt32();
                                int numLabels = brLabels.ReadInt32();

                                byte[][] pixels = new byte[28][];
                                for (int i = 0; i < pixels.Length; ++i)
                                {
                                    pixels[i] = new byte[28];
                                }

                                byte b;

                                // each image
                                for (int di = 0; di < 60000; di++)
                                {
                                    for (int i = 0; i < 28; i++)
                                    {
                                        for (int j = 0; j < 28; j++)
                                        {
                                            b = BinaryPrimitives.ReverseEndianness(brImages.ReadByte());
                                            pixels[i][j] = b;
                                        }
                                    }
                                    byte label = BinaryPrimitives.ReverseEndianness(brLabels.ReadByte());

                                    Images.Add(new Image(pixels, label));

                                }

                                return Images;
                            }
                        }
                    }
                }
            }

            //shamelessly stolen from https://jamesmccaffrey.wordpress.com/2013/11/23/reading-the-mnist-data-set-with-c/
            //modified to suit needs

        }
    }

    public class Image
    {
        public Image(byte[][] impix, byte imlbl)
        {
            Imagepixels = new byte[28][];
            for (int i = 0; i < Imagepixels.Length; ++i)
            {
                Imagepixels[i] = new byte[28];
            }
            Imagepixels = impix;
            Imagelabel = imlbl;
        }
        public byte[][] Imagepixels = new byte[28][];
        public byte Imagelabel;
    }

    public class Util
    {

        public static Random globalRandom = new Random(DateTime.Now.Millisecond);

        public void PrintNetwork()
        {

        }
    }
}
