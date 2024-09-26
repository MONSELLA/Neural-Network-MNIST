/*
ASSIGNMENT 4: NUMBER CLASSIFICATION USING ARTIFICIAL NEURAL NETWORK
AUTHOR: PAU MONSERRAT LLABRÉS
*/

package numberclassification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;

public class NumberClassification {
    //DATA
    private static ArrayList<Double[]> inputs=new ArrayList();
    private static ArrayList<Integer> labels=new ArrayList();
    
    private static ArrayList<Double[]> testX;
    private static ArrayList<Double[]> validationX;
    private static ArrayList<Double[]> trainX;
    private static ArrayList<Integer> testy;
    private static ArrayList<Integer> validationy;
    private static ArrayList<Integer> trainy;
    
    //PARAMETERS
    private static final int EPOCHS=10;
    private static final double LEARNING_RATE=0.005;
    private static final int NUM_HIDDEN_LAYERS=2;
    private static final int IMAGE_LENGTH=28*28;
            
    public static void main(String[] args) {
        //initialize the matrix inside "inputs". I need the numbers inside the matrix
        //to be flattened and double
        for(Double[] input: inputs){
            input=new Double[IMAGE_LENGTH];
        }
        
        //Read the images of the numbers from the file and store them
        try{
            File file = new File ("mnist.csv");
            FileReader fr = new FileReader (file);
            BufferedReader br = new BufferedReader(fr);
            
            //First line is not necessary
            String line = br.readLine();
            
            line = br.readLine();
            while(line!=null){
                String[] fields=line.split(",");
                
                //the first thing in each line is always the label
                labels.add(Integer.parseInt(fields[0]));
                
                //then all the data related to the number
                Double[] data=new Double[IMAGE_LENGTH];
                int index=1;
                for(int i=0;i<data.length;i++){
                    //normalize the number dividing it by 255
                    data[i]=Double.parseDouble(fields[index])/255;
                    index++;
                }
                inputs.add(data);
                
                line = br.readLine();
            }

            br.close();
        }catch(Exception e){
            System.out.println("error "+e);
        }
        
        setData();
        
        solve();
    }
    public static void setData(){
        Random r=new Random();
        Double number;
        int dataSize=labels.size();
        
        number=dataSize*0.7;
        int trainSize=number.intValue();
        number=0.1*dataSize;
        int validationSize=number.intValue();
        int testSize=dataSize-(trainSize+validationSize);
        
        //Split the data into test, validation and train sets
        testX=new ArrayList(testSize);
        validationX=new ArrayList(validationSize);
        trainX=new ArrayList(trainSize);
        testy=new ArrayList(testSize);
        validationy=new ArrayList(validationSize);
        trainy=new ArrayList(trainSize);
        
        while(trainX.size()<trainSize){
            int num=r.nextInt(dataSize);
            if(!trainX.contains(inputs.get(num))){
                trainX.add(inputs.get(num));
                trainy.add(labels.get(num));
            }
        }
        
        while(validationX.size()<validationSize){
            int num=r.nextInt(dataSize);
            if(!trainX.contains(inputs.get(num))&&!validationX.contains(inputs.get(num))){
                validationX.add(inputs.get(num));
                validationy.add(labels.get(num));
            }
        }
        
        int index=0;
        for(Double[] input: inputs){
            if(!trainX.contains(input)&&!validationX.contains(input)){
                testX.add(input);
                testy.add(labels.get(index));
            }
            index++;
        }
    }
    
    public static void solve(){
        NeuralNetwork ann=new NeuralNetwork(LEARNING_RATE,NUM_HIDDEN_LAYERS,IMAGE_LENGTH);
        //variable that stores the well predicted labels during the validation
        double bestValidationAccuracy=0;
        //variable that stores the well predicted labels during the test
        double testAccuracy=0;
        
        //Do a loop for each epoch. It will stop when the 10th epoch is reached
        //or when the accuracy for the validation set does not improve, whatever
        //comes first
        for(int epoch=0;epoch<EPOCHS;epoch++){
            //variable to store the number of well predicted labels during the training
            double trainAccuracy=0;
            //current validation accuracy
            double validationAccuracy=0;
            
            //In each epoch every image is processed once
            for(int index=0;index<trainX.size();index++){
                //train the model with every image that is in the train array
                trainAccuracy+=ann.train(trainX.get(index),trainy.get(index));
            }
            //Show the accuracy for the epoch
            System.out.println("EPOCH "+epoch+": "+(trainAccuracy/trainX.size()));

            //After each epoch, validate the model with the validation set to 
            //avoid overfitting
            for(int index=0;index<validationX.size();index++){
                validationAccuracy+=ann.validate(validationX.get(index),validationy.get(index));
            }
            
            //If the current validation accuracy is greater than the best validation
            //we have got till now, continue with one more epoch
            if(bestValidationAccuracy<validationAccuracy){
                bestValidationAccuracy=validationAccuracy;
                System.out.println("Validation accuracy: "+(validationAccuracy/validationX.size()));
                validationAccuracy=0;
            }
            //If not, break the loop to avoid overfitting
            else
                break;
        }
        
        //Evaluate the final model with the test set
        for(int index=0;index<testX.size();index++){
            testAccuracy+=ann.test(testX.get(index),testy.get(index));
        }
        //Show the final result
        System.out.println("\n"
                + "TEST ACCURACY: "+(testAccuracy/testX.size()));
    }
}
