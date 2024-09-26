
package numberclassification;

import java.util.ArrayList;
import java.util.List;

public class OutputLayer {
    int numNeurons;
    List<Neuron> neurons;
    
    public OutputLayer(int numInputs){
        //we need 10 neurons to classify each number from 0 to 9. Each neuron will
        //correspond to the probability of each one by the softmax activation function
        numNeurons=10;
        
        neurons=new ArrayList(numNeurons);
        //initialize each neuron
        for(int i=0;i<numNeurons;i++){
            Neuron newNeuron=new Neuron(numInputs);
            neurons.add(newNeuron);
        }
    }
    
    public Double[] getNets(Double[] input){
        Double[] nets=new Double[numNeurons];
        //for each neuron get the weigthed sum(nets)
        for(int i=0;i<numNeurons;i++){
            nets[i]=neurons.get(i).computeWeightedSum(input);
            //System.out.println(nets[i]);
        }
        //System.out.println("");
        return nets;
    }
    
    public static Double[] softmaxActivationFunction(Double[] nets) {
        Double[] probabilities = new Double[nets.length];
        double sum=0;
        
        //Compute exponential of the nets and sum them up
        for(int i=0;i<nets.length;i++) {
            probabilities[i]=Math.exp(nets[i]);
            //System.out.println(probabilities[i]);
            sum+=probabilities[i];
        }
        //System.out.println("");
        
        //Normalize by dividing by the sum
        for(int i=0;i<probabilities.length;i++) {
            probabilities[i]/=sum;
            //System.out.println(probabilities[i]);
        }
        //System.out.println("");
        
        return probabilities;
    }
    
    public void updateWeights(double learningRate, Double[] outputLayerErrors, Double[] input){
        //for each neuron
        for (int i=0;i<numNeurons;i++) {
            Neuron currentNeuron=neurons.get(i);
            currentNeuron.updateWeights(input, learningRate, outputLayerErrors[i]);
        }
    }
}
