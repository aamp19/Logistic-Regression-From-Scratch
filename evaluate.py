#sigmoid activation for binary classification problem
def sigmoid(x): 
    return 1.0/(1 + np.exp(-x))

#loss function used for optimization
def squared_error(y,y_hat):
    return np.square(y-y_hat)/2

#Logitic Regression model class
class LogisticRegression:
    """
    Class for building, training and evaluating the Logistic Regression model.

    -----------------------------------------------------------
    Important Attributes:
        self.weights:		  Parameters of the model that are optimized to make correct predictions on input

        self.avg_train_error:		  List containing average mean squared error per epoch

        self.epochs:          Number of times entire dataset is processed during trainin

    -----------------------------------------------------------
    Functions:
        __init__:		Initializes Logistic Regression model with randomly assigned weights, and other hyper parameres.

        train:		Takes in train set, and uses backpropagation algorithm to optimize loss function and update weights

        evaluate:		Evaluates model on test set, returning accuracy of model as decimal value

        epoch_error:		shows plot of average train and validation error per epoch

        prediction_plot:		shows plot model class prediction on test set

    """
        
    def __init__(self):
        #initialize Logitic Regression parameters and training parameters
        self.alpha= 0.01
        self.epochs=30 #change accordingly

        self.weights= np.array([random.uniform(-1,1),random.uniform(-1,1)])
        self.theta= random.uniform(-1,1)
        self.isTrained=False
        
    
    def train(self, X, y):
        
        #split train set into validation set
        self.X_train, X_valid, self.y_train, y_valid=train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.error=[]
        self.avg_train_error=[]
        self.avg_valid_error=[]

        #for each epoch, execute training steps
        for i in range(self.epochs): 
            
            for x,y in zip(self.X_train,self.y_train):

                #calculate error for each example
                z= np.dot(x,self.weights) + self.theta
                output= sigmoid(z)
                self.error.append(squared_error(y,output))

                #calculate derivatives
                dw1= -(y-output)*(sigmoid(z)*(1-sigmoid(z)))*x[0]
                dw2=  -(y-output)*(sigmoid(z)*(1-sigmoid(z)))*x[1]
                dtheta=  -(y-output)*(sigmoid(z)*(1-sigmoid(z)))

                #update weights
                self.weights[0]= self.weights[0]-self.alpha*dw1
                self.weights[1]= self.weights[1]-self.alpha*dw2
                self.theta= self.theta-self.alpha*dtheta
            
            #store average train error per epoch
            self.avg_train_error.append(np.average(self.error))
            
            
            self.error=[]

            for x,y in zip(X_valid,y_valid):

                #calculate for each example in validation set
                z= np.dot(x,self.weights) + self.theta
                output= sigmoid(z)
                self.error.append(squared_error(y,output))
            
            #store average validation error
            self.avg_valid_error.append(np.average(self.error))
            self.error=[]
            
        self.isTrained=True
            
    
    def evaluate(self, X_test,y_test):
        self.X_test= X_test
        self.y_test=y_test
        if(self.isTrained):
            #compute rounded predictions from model
            z= np.dot(X_test,self.weights) + self.theta
            output= sigmoid(z) 
            self.scores= np.round(output)
            
            #check how many algorithm got correct
            counter=0
            for yhat,y in zip(self.scores,y_test):
                if(yhat==y):
                    counter+=1
            #returns accuracy on the test set
            return counter/len(y_test)
        else:
            return
    
    
    def epoch_error(self):
        if self.isTrained:
            #plot average training and validation error per epoch
            plt.figure(1)
            plt.plot(range(self.epochs),np.array(self.avg_train_error), color="blue", label= "Average Training Error")
            plt.plot(range(self.epochs),np.array(self.avg_valid_error), color="red", label="Average Validation Error")
            plt.legend()
            plt.title("Average Error vs. Epochs")
            plt.ylabel("Average Error")
            plt.xlabel("Epochs")
            plt.show()

        
    def prediction_plot(self):
        if self.isTrained:
            #plotting test data (x1,x2, C)
            fig= plt.figure(1)
            ax= plt.axes(projection="3d")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("Y")
            ax.set_title("Test-set Actual Plot")

            #splitting test set by input features x1,x2 and class output (1 0r 0)
            temp=pd.DataFrame(self.X_test)
            temp["y"]= self.y_test
            temp["output"]= self.scores
            temp= temp.sort_values(by=['y']) #sort may not be necessary
            test_ones=np.array([i for i in temp.values[:] if i[2]==1])
            test_ones_x1= test_ones[:,0]
            test_ones_x2= test_ones[:,1]
            test_ones_y= test_ones[:,2]

            test_zeros= np.array([i for i in temp.values[:] if i[2]==0])
            test_zeros_x1= test_zeros[:,0]
            test_zeros_x2= test_zeros[:,1]
            test_zeros_y= test_zeros[:,2]
            
            ax.scatter3D(test_ones_x1, test_ones_x2, test_ones_y,"green", label="C1");
            ax.scatter3D(test_zeros_x1, test_zeros_x2, test_zeros_y,"orange", label="C2");
            ax.legend()
            
            #plotting model's prediction on test data (x1,x2, predicted class)
            fig= plt.figure(2)
            ax= plt.axes(projection="3d")
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("Yhat")
            ax.set_title("Test-set Prediction Plot")
        
            #data is split according to input features, predicted output and whether a prediction is wrong
            correct_pred=np.array([i for i in temp.values[:] if i[2]==i[3]])[:]
            correct_pred_ones=np.array([i for i in correct_pred[:] if i[2]==1])
            correct_pred_zeros=np.array([i for i in correct_pred[:] if i[2]==0])

            correct_pred_ones_x1= correct_pred_ones[:,0]
            correct_pred_ones_x2= correct_pred_ones[:,1]
            correct_pred_ones_y= correct_pred_ones[:,3]
            
            correct_pred_zeros_x1= correct_pred_zeros[:,0]
            correct_pred_zeros_x2= correct_pred_zeros[:,1]
            correct_pred_zeros_y= correct_pred_zeros[:,3]
            
            
            wrong_pred= np.array([i for i in temp.values[:] if i[2]!=i[3]])[:]
            wrong_pred_x1= wrong_pred[:,0] 
            wrong_pred_x2= wrong_pred[:,1] 
            wrong_pred_y= wrong_pred[:,3]

            ax.scatter3D(correct_pred_ones_x1, correct_pred_ones_x2,correct_pred_ones_y, "green", label="C1")
            ax.scatter3D(correct_pred_zeros_x1, correct_pred_zeros_x2, correct_pred_zeros_y, "orange", label="C2")
            ax.scatter3D(wrong_pred_x1,wrong_pred_x1, wrong_pred_y, "red", label="WRONG")
            ax.legend()
        
