
package numberclassification;

import java.util.Random;

public class Neuron {
    private final double ALPHA=0.05;
    private double[] weights;
    private double output;
    
    public Neuron(int numInputs){
        Random r=new Random();
        
        //Initialize weights
        weights=new double[numInputs];
        for(int i=0;i<weights.length;i++){
            weights[i]=r.nextDouble(-0.5,0.5);
        }
    }
    
    public double[] getWeights(){
        return weights;
    }
    
    public double getOutput(){
        return output;
    }
    
    public double computeOutput(Double[] inputs){
        //compute the output applying the activation function to the weighted sum
        return ReLUActivationFunction(computeWeightedSum(inputs));
    }
    
    public double computeWeightedSum(Double[] inputs){
        double weightedSum=0;
        for (int i=0;i<inputs.length;i++) {
            weightedSum+=inputs[i]*weights[i];
        }
        return weightedSum;
    }
    
    private double ReLUActivationFunction(double weightedSum) {
        output=Math.max(0,weightedSum);
        return output;
    }
    
    public double ReLUDerivative() {
        return output>0 ? 1:0;
    }
    
    public void updateWeights(Double[] input, double learningRate, Double error) {
        for (int i=0;i<weights.length;i++) {
            // Update the weight using gradient descent
            weights[i]-=learningRate*error*input[i];
            //System.out.println(input[i]);
        }
        //System.out.println("");
    }
}
