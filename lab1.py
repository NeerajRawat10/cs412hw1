import torch
import sklearn, sklearn.datasets, sklearn.model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt


# ======================================================================================================
# Learner
# ======================================================================================================

class LinearPotentials(torch.nn.Module):
    def __init__(self, input_dim, output_dim, random_weights_init = True):
        super(LinearPotentials, self).__init__()
        self.num_features = input_dim
        self.num_classes = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.history = {'train_loss':[], 'test_loss': [], 'accuracy':[]}
        if random_weights_init == True:
            self.linear.weight.data.uniform_(0.0, 1.0)
            self.linear.bias.data.fill_(0)
            
    def forward(self, x):
        outputs = self.linear(x)
        return outputs
    
    def plot_learning_curves(self, title=''):
        title = 'Learning Curves Plot' if title == '' else title
        fig = plt.figure(figsize=(14, 4))
        iters = np.arange(0, len(self.history['train_loss']))
        plt.plot(iters, self.history['train_loss'], linestyle='dashed',  label = 'Train Loss')
        plt.plot(iters, self.history['test_loss'],  linestyle='-',  label = 'Test Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.title(title)
        #plot(iters, self.history['train_loss'])
        plt.show()
        return fig
    
    def compute_error_stats(self):
        ''' DESCRIPTION: This class function computes the mean and std values from logged history
                         Make sure all test_accuracy, train_loss, test_loss are of equal size
        '''
        mean, std = 0,0

        mean_test_accuracy = np.mean(self.history['accuracy'])
        mean_train_loss = np.mean(self.history['train_loss'])
        mean_test_loss = np.mean(self.history['test_loss'])
        std_test_accuracy = np.std(self.history['accuracy'])
        print("Mean Accuracy: {}, std: {}".format(mean_test_accuracy, std))
        return mean_test_accuracy, std

class OutputHook(list):
    """ Hook to capture module outputs. Used For the L1/2 regularization part
    """
    def __call__(self, module, input, output):
        self.append(output)
        
# ======================================================================================================
# Functions
# ======================================================================================================


def plot(x_index, y_index, data):
    formatter = plt.FuncFormatter(lambda i, *args: data.target_names[int(i)])
    plt.scatter(data.data[:, x_index], data.data[:, y_index], c=data.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(data.feature_names[x_index])
    plt.ylabel(data.feature_names[y_index])
# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------
def evaluate_model(model, data, optimizer, lr = 0.1, criterion = torch.nn.CrossEntropyLoss(), 
                   number_of_epochs = 10000, print_interval = 100, debug= False):
        
    # Handle Inputs
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    # ---|
    
    # YOUR CODE HERE
    # ??????????????
    for epoch in range(number_of_epochs): 
        y_prediction=model(x_train)          # make predictions
        loss=criterion(y_prediction,y_train) # calculate losses
        model.history['train_loss'].append(loss.item())
        loss.backward()                      # obtain gradients
        optimizer.step()                     # update parameters
        optimizer.zero_grad()                # reset gradients
    
        
        y_prob = torch.softmax(model(x_test), 1)
        y_pred = torch.argmax(y_prob, axis=1)

        train_log_loss = criterion(model(x_train), y_train).detach().numpy()
        test_log_loss = criterion(model(x_test), y_test).detach().numpy()
        test_accuracy = (sum(y_pred==y_test)/y_test.shape[0]).detach().numpy()
        
        model.history['test_loss'].append(test_log_loss.item())
        model.history['accuracy'].append(test_accuracy)
        if (epoch+1)%print_interval == 0:
            print('Epoch:', epoch+1,',loss=',loss.item())

    # Print model parameters
    if debug == True:
        for param in model.named_parameters():
            print("Param = ",param)     
    
    # ???|
    print("Train Log Loss = ", train_log_loss)
    print("Test Log Loss  = ", test_log_loss)
    print("Test Accuracy  = ", test_accuracy) 
    model.plot_learning_curves()
    # Do not change return types.
    return train_log_loss, test_log_loss, test_accuracy

# -----------------------------------------------------------------------------------------------
def learning_rate_iter_explore(model, data, optimizer, lr_range = [0.001, 0.1, 0.25, 0.5],
                               criterion = torch.nn.CrossEntropyLoss(), number_of_epochs = 10000, 
                               print_interval = 1000):
    ''' DESCRIPTION: This function should explore the trade off between learning rate and iterations of training.
                     It should iteratively train a model on a given learing rate and mark the best performance on the test
                     set for a given number of iterations. It should report the best learning rate-iteration combo.
                     
        ARGUMENTS: model (nn module): Learner model. nn module type
                   data (dictionary):  The train/test data provided a dictionary 
                                       data = {'x_train':torch.tensor, 'x_test': torch.tensor, 'y_train':torch.tensor, 'y_test': torch.tensor}  
                   optimizer(nn optim): Chose optimizer, i.e adam, SGD etc
    
                   lr_range (list): list containg the learning rates to be explored.
                   criterion (nn loss): NN loss function. i.e CrossEntropyLoss.
                   number_of_epochs (int): How many epochs each model should be evaluated for.
                   print_interval (int):   Per how many iters should the script print out info.
                   
        RETURNS: best_lr (float)
                 best_iters (int)
    '''
    
    best_lr, best_iters = 0,0
    resultsList = []
    resultsDict = {'train_log_loss':[], 'test_log_loss':[], 'test_accuracy':[], 'learning_rates':[]}
    
    for i, lr in enumerate(lr_range):
        # Create a copy of the original model, with the same weights and evaluate it.
        cur_model = LinearPotentials(model.num_features, model.num_classes)
        cur_model.load_state_dict(model.state_dict().copy())
        print("Evaluating Model with learning rate {}; Starting params: {}".format(lr, cur_model.linear.weight))
        curr_optim = type(optimizer)(cur_model.parameters(),lr=lr) # torch.optim.Adam(model.parameters(), lr=0.1)
        train_log_loss, test_log_loss, test_accuracy = evaluate_model(cur_model, data, curr_optim, 
                                                                      criterion = torch.nn.CrossEntropyLoss(),
                                                                      print_interval=print_interval, number_of_epochs = number_of_epochs)
        resultsDict['train_log_loss'].append(float(train_log_loss))
        resultsDict['test_log_loss'].append(float(test_log_loss))
        resultsDict['test_accuracy'].append(float(test_accuracy))
        resultsDict['learning_rates'].append(float(lr))
        resultsList.append([float(train_log_loss), float(test_log_loss), float(test_accuracy)])
        
    return resultsDict

# -----------------------------------------------------------------------------------------------
def evaluate_svm_model(data, criterion = torch.nn.CrossEntropyLoss(), 
                   number_of_epochs = 10000, print_interval = 1000):
    ''' DESCRIPTION: This function should fit an SVM on the train data and targets and test it
                     on the Y data and targets. Report accuracy on test set
                     
        RETURNS test_accuracy (float)
                
    '''
    train_log_loss, test_log_loss, test_accuracy = 0,0,0
    
    # Handle Inputs
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    # ---|
    
    # YOUR CODE HERE
    # ??????????????
    # Instantiate an SVM based on libsvm
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    print("\nFitting SVM to training data!!")
    print("====================================")
    # Train model on train data
    clf.fit(x_train, y_train)
    # Need to transform output to PyTorch tensor (SM returns ndarray)
    # ???|
    
    y_pred = torch.tensor(clf.predict(x_test))    
    test_accuracy = (sum(y_pred==y_test)/y_test.shape[0]).detach().numpy()   # get acc
    print("SVM Test Accuracy  = ", test_accuracy)
       
    return test_accuracy

# -----------------------------------------------------------------------------------------------

def evaluate_with_model_regularization(model, data, optimizer, criterion = torch.nn.CrossEntropyLoss(), 
                                       reg_lambda = 0.01, reg_type = 'l1',
                                       number_of_epochs = 30000, print_interval = 1000, debug = False):
        
    # Handle Inputs
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    # ---|
    
    output_hook = OutputHook()
    model.linear.register_forward_hook(output_hook)

    # YOUR CODE HERE
    # ??????????????
    for epoch in range(number_of_epochs): 
        y_prediction=model(x_train)          # make predictions
        loss=criterion(y_prediction,y_train) # calculate losses

        # Compute the L1 and L2 penalty of parameters and add to the loss
        # YOUR CODE HERE 

        loss.backward()                      # obtain gradients
        optimizer.step()                     # update parameters
        optimizer.zero_grad()                # reset gradients
        if (epoch+1)%print_interval == 0:
            print("Epoch: {}, {}-loss: {}".format(epoch+1,reg_type, loss.item()))
        
        # Get predictions on the test set
        y_prob = torch.softmax(model(x_test), 1)
        y_pred = torch.argmax(y_prob, axis=1)
        train_log_loss = loss
        test_log_loss = (criterion(model(x_test), y_test) +l_penalty).detach().numpy()
        test_accuracy = (sum(y_pred==y_test)/y_test.shape[0]).detach().numpy()
        # Log all loss progress
        model.history['train_loss'].append(loss.item())
        model.history['test_loss'].append(test_log_loss.item())
        model.history['accuracy'].append(test_accuracy)
        
    # Print model parameters
    if debug == True:
        for param in model.named_parameters():
            print("Param = ",param) 
 
    fig = model.plot_learning_curves()

    # ???|
    print("Train Log Loss = ", train_log_loss)
    print("Test Log Loss  = ", test_log_loss)
    print("Test Accuracy  = ", test_accuracy) 
    
    # Do not change return types.
    return train_log_loss, test_log_loss, test_accuracy

# ======================================================================================================
# MAIN
# ======================================================================================================
def main():
    
    # Part 0
    # Load and visualize data.
    iris = sklearn.datasets.load_iris()
    x,y = iris.data,iris.target

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    plot(0, 1, iris)
    plt.subplot(122)
    plot(2, 3, iris)
    plt.show()

    
    # split data into training and testing sets
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

    # convert to tensors for Pytorch
    x_train = torch.from_numpy(x_train.astype(np.float32))
    x_test  = torch.from_numpy(x_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.int_))
    y_test  = torch.from_numpy(y_test.astype(np.int_))
    data    = dict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # Get data info
    num_features, num_classes = x_train.shape[1], y.max()+1
    # ---|
    
    # Part 1
    # Declare learner model, loss objective, optimizer
    # Initialize learner with random weights
    model = LinearPotentials(num_features, num_classes, random_weights_init = True)
    criterion = torch.nn.CrossEntropyLoss() # torch.nn.MultiMarginLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1, momentum=0.1) # torch.optim.Adam(model.parameters(), lr=0.1)
    # ---|
    
    # Part 2
    # ------------
    # Given a learning rate of 0.1 train on train data and report accuracy on test data
    # Func2_call 
    evaluate_model(model, data, optimizer, criterion = torch.nn.CrossEntropyLoss(), number_of_epochs = 100)
    model.compute_error_stats()
    # ---|
    
    # Part 3
    # ------------
    # Evaluate 100 instances of the model above and report mean error rate and error std.
    # Func3_call 
    # ---|
    
    # Part 4 
    # A. Repeat steps 2,3 while adding L1 regularization to the objective
    # Func4_call 
    #evaluate_with_model_regularization(model, data, optimizer, criterion = torch.nn.CrossEntropyLoss(), number_of_epochs = 1000)
    # B. Repeat steps 2,3 while adding L2 regularization to the objective
    # ---|
    
    # Part 5
    # Explore different learning rates. For a given learning rate collection, report the lr-vs-iteration tradefoff, That is
    # for each learning rate, plot the test loss vs accuracy and report on where you believe the training has converged to a solution.
    # Func5_call 
    #learning_rate_iter_explore(model, data, optimizer, criterion = torch.nn.CrossEntropyLoss(), number_of_epochs = 1000)
    
    # Part 6
    # SVM deployment. Using sklearn's SVM implmentation, train an SVM with a radial basis function kernel on the training set and report 
    # its performance in terms of Accuracy on the test set
    # # Func6_call 
    #evaluate_svm_model(data)
if __name__ == "__main__":
    main()
