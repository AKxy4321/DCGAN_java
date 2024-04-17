package cnn;

public interface Layer {
     Tensor forwardProp(Tensor input);
     Tensor backProp(Tensor outputGrad, double learningRate);
     int[] getOutputDims();
}