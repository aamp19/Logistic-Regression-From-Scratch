Logistic regression is a statistical model which uses a logistic function to model a binary 
dependent variable. Logistic Regression displays the probabilites for classification problem
where there are two potential outcomes. Unlike Linear Regression the logic regression model 
does not determine future values on a linear scale. It uses a logistic function which is more
realistic to use rather than a linear regression model for real life problems. 

For this problem I imported a dataset into the program, and split the dataset into train and 
test data. I then created a train function which trains the training set using gradient descent
on the sigmoid function (cost function). Once this was done I was able to determine if there was
any errors based on how the graph was plotted based on the values I generated from the prediction 
function I created vs. the true values of the dataset