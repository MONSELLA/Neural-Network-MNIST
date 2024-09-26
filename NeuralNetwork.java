
package numberclassification;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final int NEURONS_IN_EACH_LAYER=256;
    private double learningRate;
    
    ArrayList<HiddenLayer> hiddenLayers;
    OutputLayer outputLayer;
    
    public NeuralNetwork(double learningRate, int numHiddenLayers, int numInputs){
        //initialzie the hidden layers
        hiddenLayers=new ArrayList(numHiddenLayers);
        for(int i=0;i<numHiddenLayers;i++){
            if(i==0){
                //The number of inputs of the first hidden layer are the 28*28 pixels
                //that every image has
                hiddenLayers.add(new HiddenLayer(NEURONS_IN_EACH_LAYER, numInputs));
            }else{
                //The number of inputs of the other hidden layers are the outputs
                //of the last layer's neurons
                hiddenLayers.add(new HiddenLayer(NEURONS_IN_EACH_LAYER, NEURONS_IN_EACH_LAYER));
            }
        }
        
        //initialize the output layer
        outputLayer=new OutputLayer(NEURONS_IN_EACH_LAYER);
        
        this.learningRate=learningRate;
    }
    
    public int train(Double[] tX, int ty){
        //Do forward propagation till we get the outputs from the output layer
        Double[] outputLayerOutput=forwardStep(tX);
        
        //Apply softmax activation function to output layer output to get the 
        //probabilities for each number
        Double[] predictedProbabilities = outputLayer.softmaxActivationFunction(outputLayerOutput);
        
        //Get the predicted label
        int predictedLabel=getPredictedLabel(predictedProbabilities);
        //System.out.println(predictedLabel);
        
        //Compute error of the output layer
        Double[] OLerror=computeOutputLayerError(ty,predictedProbabilities);
        /*
        for(int i=0;i<OLerror.length;i++){
            System.out.println(OLerror[i]);
        }
        System.out.println("");
        */
        
        //Do backpropagation
        ArrayList<Double[]> HLerrors=backwardStep(OLerror);
        
        //Update weights
        updateWeights(HLerrors,OLerror,tX);
        
        return predictedLabel==ty ? 1:0;
    }
    
    private Double[] forwardStep(Double[] tX){
        Double[] outputs=new Double[NEURONS_IN_EACH_LAYER];
        //Do the forward step for each hidden layer
        for(int index=0;index<hiddenLayers.size();index++){
            //The first layer is the one that receives the inputs
            if(index==0)
                outputs=hiddenLayers.get(index).forwardPropagation(tX);
            //The others receives the outputs from the last layer
            else
                outputs=hiddenLayers.get(index).forwardPropagation(outputs);
        }
        
        return outputLayer.getNets(outputs);
    }
    
    private Double[] computeOutputLayerError(int ty,Double[] predictedProbabilities){
        int[] trueProbabilities=new int[10];
        trueProbabilities[ty] = 1; //One-hot encoding of the true label
        
        return computeError(predictedProbabilities, trueProbabilities);
    }
    
    private Double[] computeError(Double[] predictedProbabilities, int[] trueProbabilities) {
        Double[] error = new Double[10];
        for (int i=0;i<predictedProbabilities.length;i++) {
            error[i]=predictedProbabilities[i]-trueProbabilities[i];
        }
        return error;
    }
    
    private ArrayList<Double[]> backwardStep(Double[] outputLayerErrors){
        //an array list that has the array of the errors for each neuron of the layer
        ArrayList<Double[]> HLerrors=new ArrayList(hiddenLayers.size());
        //this has the errors from the next layer
        Double[] errorFromNextLayer=outputLayerErrors.clone();
        
        //for each hidden layer calculate the error for each neuron
        for(int HLnumber=hiddenLayers.size()-1;HLnumber>=0;HLnumber--){
            //get the current hidden layers
            HiddenLayer currentHL=hiddenLayers.get(HLnumber);
            //get the array of errors for each hidden layer
            Double[] errors=currentHL.backwardPropagation(errorFromNextLayer);
            //add the errors array to the arraylist
            HLerrors.add(errors);
            //update the errorFromNextLayer
            errorFromNextLayer=errors;
        }
        return HLerrors;
    }
    
    private int getPredictedLabel(Double[] predictedProbabilities){
        int predictedLabel=0;
        double probPredictedLabel=Double.NEGATIVE_INFINITY;
        
        for(int i=0;i<predictedProbabilities.length;i++){
            if(predictedProbabilities[i]>probPredictedLabel){
                predictedLabel=i;
                probPredictedLabel=predictedProbabilities[i];
            }
        }
        
        return predictedLabel;
    }
    
    private void updateWeights(ArrayList<Double[]> hiddenLayerErrors, Double[] outputLayerErrors,Double[] input){
        //update the weights
        Double[] input_copy=input.clone();
        for(int index=0;index<hiddenLayers.size();index++){
            //get the ith hiddenlayer
            HiddenLayer currentHL=hiddenLayers.get(index);
            input_copy=currentHL.updateWeights(learningRate,hiddenLayerErrors.get(index),input_copy);
        }
        outputLayer.updateWeights(learningRate,outputLayerErrors,input_copy);
    }
    
    public int validate(Double[] vX, int vy){
        //Do forward propagation till we get the outputs from the output layer
        Double[] outputLayerOutput=forwardStep(vX);
        
        //Apply softmax activation function to output layer output to get the 
        //probabilities for each number
        Double[] predictedProbabilities = outputLayer.softmaxActivationFunction(outputLayerOutput);
        
        //Get the predicted label
        int predictedLabel=getPredictedLabel(predictedProbabilities);
        
        //Return 1 if it is well predicted and 0 otherwise
        return predictedLabel==vy ? 1:0;
    }
    
    public int test(Double[] testX, int testy){
        //Do forward propagation till we get the outputs from the output layer
        Double[] outputLayerOutput=forwardStep(testX);
        
        //Apply softmax activation function to output layer output to get the 
        //probabilities for each number
        Double[] predictedProbabilities = outputLayer.softmaxActivationFunction(outputLayerOutput);
        
        //Get the predicted label
        int predictedLabel=getPredictedLabel(predictedProbabilities);
        
        //Return 1 if it is well predicted and 0 otherwise
        return predictedLabel==testy ? 1:0;
    }
}
