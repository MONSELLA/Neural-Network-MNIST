
package numberclassification;

import java.util.ArrayList;
import java.util.List;

public class HiddenLayer {
    int numNeurons;
    List<Neuron> neurons;
    
    public HiddenLayer(int numNeurons, int numInputs){
        this.numNeurons=numNeurons;
        neurons=new ArrayList(numNeurons);
        //initialize each neuron
        for(int i=0;i<numNeurons;i++){
            Neuron newNeuron=new Neuron(numInputs);
            neurons.add(newNeuron);
        }
    }
    
    public Double[] forwardPropagation(Double[] tX){
        Double[] outputs=new Double[numNeurons];
        //for each neuron compute the output
        for(int i=0;i<numNeurons;i++){
            outputs[i]=neurons.get(i).computeOutput(tX);
        }
        /*
        for(int i=0;i<outputs.length;i++){
            System.out.println(outputs[i]);
        }
        System.out.println("");
        */
        return outputs;
    }
    
    public Double[] backwardPropagation(Double[] errorFromNextLayer) {
        //stores the errors of each neuron from this layer
        Double[] error=new Double[numNeurons];
        
        //for each neuron
        for (int i=0;i<numNeurons;i++) {
            double neuronError=0;
            Neuron currentNeuron=neurons.get(i);
            
            double[] weights=currentNeuron.getWeights();
            for (int j=0;j<errorFromNextLayer.length;j++) {
                // Compute the error contribution for the current neuron
                neuronError += errorFromNextLayer[j]*weights[j];
            }
            error[i]=neuronError;
            //multiply with the derivate of the activation function of the layer
            error[i]*=currentNeuron.ReLUDerivative();
        }
        return error;
    }
    
    public Double[] updateWeights(double learningRate, Double[] hiddenLayerErrors,Double[] input){
        //Make an array for the outputs of this neurons
        Double[] outputs=new Double[numNeurons];
        
        //for each neuron
        for (int i=0;i<numNeurons;i++) {
            Neuron currentNeuron=neurons.get(i);
            currentNeuron.updateWeights(input, learningRate, hiddenLayerErrors[i]);
            outputs[i]=currentNeuron.getOutput();
        }
        return outputs;
    }
}
