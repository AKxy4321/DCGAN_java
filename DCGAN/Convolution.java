package DCGAN;

import java.util.Arrays;

public class Convolution {
    public int numFilters;
    public int filterSize;
    public int stride;
    public double[][][][] filters;
    public double[] biases;
    public double[][][] input3D;
    public int output_width, output_height, output_depth;
    public int input_width, input_height, input_depth;

    public int filter_depth;

    Convolution(int filterSize, int numFilters, int stride, int input_width, int input_height, int input_depth) {
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.stride = stride;
        this.input_width = input_width;
        this.input_height = input_height;
        this.input_depth = input_depth;

        this.output_width = (int) Math.floor((double) (input_width - filterSize) / stride + 1);
        this.output_height = (int) Math.floor((double) (input_height - filterSize) / stride + 1);
        this.output_depth = this.numFilters;

        this.filter_depth = input_depth;

//        this.filters = new double[numFilters][input_depth][filterSize][filterSize];
//        this.biases = new double[numFilters];

        this.filters = XavierInitializer.xavierInit4D(numFilters, filter_depth, filterSize);

        this.biases = XavierInitializer.xavierInit1D(numFilters);
    }


    public double[][][] forward(double[][][] input) {
        this.input3D = input;

        double[][][] output = new double[output_depth][output_height][output_width];

        for (int d = 0; d < this.output_depth; d++) {
            double[][][] f = this.filters[d];

            int y = 0; //-this.padding;
            for (int output_y = 0; output_y < this.output_height; y += stride, output_y++) {  // xy_stride
                int x = 0; // -this.padding;
                for (int output_x = 0; output_x < this.output_width; x += stride, output_x++) {  // xy_stride

                    // convolve centered at this particular location
                    double a = 0.0;
                    for (int fy = 0; fy < filterSize; fy++) {
                        int input_y = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < filterSize; fx++) {
                            int input_x = x + fx;
                            if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                                for (int fd = 0; fd < f.length; fd++) {
                                    a += f[fd][fy][fx] * input[fd][input_y][input_x];
                                }
                            }
                        }
                    }
                    a += this.biases[d];
                    output[d][output_y][output_x] = a;
                }
            }
        }

        return output;
    }

    public double[][][] backprop(double[][][] output_gradients) {

        double[][][] inputGradient = new double[input_depth][input_height][input_width];


        for (int d = 0; d < this.output_depth; d++) {
            double[][][] f = this.filters[d];

            int y = 0; //-this.padding;
            for (int output_y = 0; output_y < this.output_height; y += stride, output_y++) {
                int x = 0; // -this.padding;
                for (int output_x = 0; output_x < this.output_width; x += stride, output_x++) {

                    // convolve centered at this particular location
                    double chain_grad = output_gradients[d][output_y][output_x];
                    for (int fy = 0; fy < filterSize; fy++) {
                        int input_y = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < filterSize; fx++) {
                            int input_x = x + fx;
                            if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                                for (int fd = 0; fd < f.length; fd++) {
                                    inputGradient[fd][input_y][input_x] += f[fd][fy][fx] * chain_grad;

                                }
                            }
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    public void updateParameters(double[][][] output_gradients, double learning_rate) {

        double[][][][] filterGradients = new double[numFilters][filter_depth][filterSize][filterSize];
        double[] biasGradients = new double[numFilters];

        for (int d = 0; d < this.output_depth; d++) {
            double[][][] f = this.filters[d];

            int y = 0; //-this.padding;
            for (int output_y = 0; output_y < this.output_height; y += stride, output_y++) {
                int x = 0; // -this.padding;
                for (int output_x = 0; output_x < this.output_width; x += stride, output_x++) {

                    // convolve centered at this particular location
                    double chain_grad = output_gradients[d][output_y][output_x];
                    for (int fy = 0; fy < filterSize; fy++) {
                        int input_y = y + fy; // coordinates in the original input array coordinates
                        for (int fx = 0; fx < filterSize; fx++) {
                            int input_x = x + fx;
                            if (input_y >= 0 && input_y < input_height && input_x >= 0 && input_x < input_width) {
                                for (int fd = 0; fd < f.length; fd++) {
                                    filterGradients[d][fd][fy][fx] += input3D[fd][input_y][input_x] * chain_grad;
                                }
                            }
                        }
                    }
                    biasGradients[d] += chain_grad;
                }
            }
        }

        for (int k = 0; k < filters.length; k++) {
            for (int fd = 0; fd < filters[0].length; fd++) {
                for (int fy = 0; fy < filterSize; fy++) {
                    for (int fx = 0; fx < filterSize; fx++) {
                        this.filters[k][fd][fy][fx] -= learning_rate * filterGradients[k][fd][fy][fx];
                    }
                }
            }
            this.biases[k] -= learning_rate * biasGradients[k];
        }

    }


    public static void test(double[][][] input, double[][][] filter){
        Convolution conv = new Convolution(filter[0].length, 1, 1, input[0][0].length, input[0].length, input.length);
        conv.filters[0] = filter;
        conv.biases[0] = 0.0;
        System.out.println("convolution output:");
        UTIL.prettyprint(conv.forward(input));
    }
    public static void main(String[] args) {
        Convolution conv = new Convolution(2, 1, 1, 4, 4, 1);
        double[][][] input = new double[][][]{{{0.30817199884474633, 0.46007586122875466, 0.21450701335303385, 0.11559490853249392},
                {0.20504174993982818, 0.19859653628612528, 0.018425067092649523, 0.020424015094034804},
                {0.09712189525746204, 3.375814623831309E-4, 0.032548058921766605, 0.01273730366496365},
                {0.171363639635987, 0.0630113410210414, 0.021608778257684216, 0.09606253244241483}}};
        conv.filters[0][0] = new double[][]{{-0.5624438919050452, -0.34838378252869157 , 0.04610379248030185},
                {0.28937128145669205, 0.3686168439781439 , 0.5574219252868703},
                {-0.1631085894638904, 0.4608589364400626 , 0.1311166708761941}};
        conv.biases[0] = 0.0;
        /*
        * outputGradient
{0.05317230973804788, 0.059228707518354434, 0.0717510057086238, 0.06506861852353718},
{0.23209021166063673, 0.13446081861270737, 0.01012569918377766, 2.0856825422670703E-4},
{0.12118979940981606, 0.005367829964930401, 0.20792565550237252, 0.032808395675822014},
{0.009400669754009633, 0.00250540600013088, 0.007655959853797619, 0.003541766208192948}
*
*
*
{{0.005935862539757215, 0.014767443339451937 ,0.009909938388943176 ,0.00303290895296782 },
{0.06920737939169946, 0.04120547600111788 ,0.0010202652801449734 ,0.008468274157925167 },
{0.016703371369955897, 1.822085349390881E-4 ,0.005620206829124786 ,0.04739678470843118 },
{0.003224749972755033, 0.009018999713874725 ,0.020327664981810344 ,0.03267296750775934 },}
InputGradient
-0.005467486424117771 0.017382150242758028
0.029815899652575212 0.042605585788895166
input
1.0 1.0
1.0 1.0
filterGradient
0.4789520475297464 0.2755662310234633 0.14715389167016535
0.4931086596480906 0.35788000326378794 0.2510683186161989
0.13846370512888698 0.2234548513212314 0.25193177724018506
*
*
*
* paddedOutput
{{0.30817199884474633, 0.46007586122875466, 0.21450701335303385, 0.11559490853249392}
{0.20504174993982818, 0.19859653628612528, 0.018425067092649523, 0.020424015094034804}
{0.09712189525746204, 3.375814623831309E-4, 0.032548058921766605, 0.01273730366496365}
{0.171363639635987, 0.0630113410210414, 0.021608778257684216, 0.09606253244241483}}
InputGradient
0.3963510184628816 0.24525358098179723
0.15289114590001063 -0.011571234609882429
Filters
{{-0.5624438919050452, -0.34838378252869157 , 0.04610379248030185}
{0.28937128145669205, 0.3686168439781439 , 0.5574219252868703}
{-0.1631085894638904, 0.4608589364400626 , 0.1311166708761941}}
        * */

        UTIL.prettyprint(conv.forward(input));
    }

}