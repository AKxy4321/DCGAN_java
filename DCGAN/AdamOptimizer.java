package DCGAN;

public class AdamOptimizer {
    private double learningRate;
    private double beta1;
    private double beta2;
    private double epsilon;
    private double m;
    private double v;
    private int t;

    public AdamOptimizer(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.m = 0;
        this.v = 0;
        this.t = 0;
    }

    public double update(double gradient) {
        this.t++;
        this.m = this.beta1 * this.m + (1 - this.beta1) * gradient;
        this.v = this.beta2 * this.v + (1 - this.beta2) * Math.pow(gradient, 2);
        double mHat = this.m / (1 - Math.pow(this.beta1, this.t));
        double vHat = this.v / (1 - Math.pow(this.beta2, this.t));
        double stepSize = this.learningRate / (this.epsilon + Math.sqrt(vHat));
        return mHat * stepSize;
    }
}
