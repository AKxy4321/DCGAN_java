package DCGAN;

public class LeakyReLULayer {
    
    
    public float[][][] input;
    public float[][][] output;

    public float[][] input2D;
    public float[][] output2D;
    float k;


    public LeakyReLULayer(){
        this.k = 0.01f;
    }

    public LeakyReLULayer(float k){
        this.k =k;
    }

    public float apply_leaky_relu(float x){
        return x *k;
    }

    public float[][][] forward(float[][][] input) {
        this.input = input;
        int depth = input.length;
        int height = input[0].length;
        int width = input[0][0].length;
        output = new float[depth][height][width];

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output[d][h][w] = apply_leaky_relu(input[d][h][w]);
                }
            }
        }

        return output;
    }

    public float[][][] forward(float[][] input) {
        this.input2D = input;
        int height = input.length;
        int width = input[0].length;
        output2D = new float[height][width];

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    output2D[h][w] = apply_leaky_relu(input[h][w]);
                }
            }
        return output;
    }

    public float[][][] backprop(float[][][] d_L_d_out) {
        float[][][] d_L_d_input = new float[d_L_d_out.length][d_L_d_out[0].length][d_L_d_out[0][0].length];
        int depth = d_L_d_out.length;
        int height = d_L_d_out[0].length;
        int width = d_L_d_out[0][0].length;

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // If the input value was negative, the gradient is k; 
                    d_L_d_input[d][h][w] = input[d][h][w] > 0 ? d_L_d_out[d][h][w] : k*d_L_d_out[d][h][w];
                }
            }
        }
        return d_L_d_input;
    }

    public float[][] backprop(float[][] d_L_d_out) {
        float[][] d_L_d_input = new float[d_L_d_out.length][d_L_d_out[0].length];
        int height = d_L_d_out.length;
        int width = d_L_d_out[0].length;

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // If the input value was negative, the gradient is k;
                    d_L_d_input[h][w] = input2D[h][w] > 0 ? d_L_d_out[h][w] : k*d_L_d_out[h][w];
                }
            }
        return d_L_d_input;
    }

}