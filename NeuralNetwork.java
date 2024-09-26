
package numberclassification;

public class NeuralNetwork {
    private final int NUM_NEURONS=256;
    private double learningRate;
    
    HiddenLayer hiddenLayer;
    OutputLayer outputLayer;
    
    public NeuralNetwork(double learningRate, int numInputs){
        //The number of inputs of the hidden layer are the 28*28 pixels that every image has
        hiddenLayer=new HiddenLayer(NUM_NEURONS, numInputs);
        
        //initialize the output layer
        outputLayer=new OutputLayer(NUM_NEURONS);
        
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
        
        //Compute error of the output layer
        Double[] OLerror=computeOutputLayerError(ty,predictedProbabilities);
        
        //Do backpropagation
        Double[] HLerrors=backwardStep(OLerror);
        
        //Update weights
        updateWeights(HLerrors,OLerror,tX);
        
        //Return the predicted label
        return predictedLabel;
    }
    
    private Double[] forwardStep(Double[] data){
        //The hidden layer is the one that receives the inputs
        Double[] outputs=hiddenLayer.forwardPropagation(data);
        
        return outputLayer.getNets(outputs);
    }
    
    private Double[] computeOutputLayerError(int labels, Double[] predictedProbabilities){
        int[] trueProbabilities=new int[10];
        //One-hot encoding of the true label
        trueProbabilities[labels] = 1;
        
        return computeError(predictedProbabilities, trueProbabilities);
    }
    
    private Double[] computeError(Double[] predictedProbabilities, int[] trueProbabilities) {
        Double[] error = new Double[10];
        for (int i=0;i<predictedProbabilities.length;i++) {
            error[i]=predictedProbabilities[i]-trueProbabilities[i];
        }
        return error;
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
    
    private Double[] backwardStep(Double[] outputLayerErrors){
        //get the array of errors for the hidden layer and return them
        return hiddenLayer.backwardPropagation(outputLayerErrors);
    }
    
    private void updateWeights(Double[] hiddenLayerErrors, Double[] outputLayerErrors, Double[] input){
        //variable to store the outputs from the hidden layer
        Double[] hiddenLayerOutputs;
        //update the weights of the hidden layer
        hiddenLayerOutputs=hiddenLayer.updateWeights(learningRate,hiddenLayerErrors,input);
        //update the weights of the output layer
        outputLayer.updateWeights(learningRate,outputLayerErrors,hiddenLayerOutputs);
    }
    
    public int test(Double[] testX){
        //Do forward propagation till we get the outputs from the output layer
        Double[] outputLayerOutput=forwardStep(testX);
        
        //Apply softmax activation function to output layer output to get the 
        //probabilities for each number
        Double[] predictedProbabilities = outputLayer.softmaxActivationFunction(outputLayerOutput);
        
        //Get the predicted label
        int predictedLabel=getPredictedLabel(predictedProbabilities);
        
        //Return the predicted label
        return predictedLabel;
    }
}
