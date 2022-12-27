import numpy
import scipy.special # for sigmoid function

# neural network class definition
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden and output layer
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        # learning rate
        self.lr=learningrate

        # weights
        seed = 2
        numpy.random.seed(seed)
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        numpy.random.seed(seed+10)
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        #activation function
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        # calculate singlas emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate singals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error
        output_errors = targets - final_outputs
        # hidden layer error
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layer
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and the hidden layer
        #self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        self.wih += self.lr * numpy.dot(numpy.transpose(numpy.dot(numpy.transpose(output_errors * final_outputs * (1.0 - final_outputs)), self.who) * numpy.transpose(hidden_outputs * (1.0 - hidden_outputs))),numpy.transpose(inputs))
        
        pass

    # query the neural network
    def query(self, input_list):
        #convert inputs list to 2d array
        inputs=numpy.array(input_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        #calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
epochs = 1

# learning rate
learning_rate = 0.2

# create instance of neural network
n = neuralNetwork(inputnodes=input_nodes, hiddennodes=hidden_nodes, outputnodes=output_nodes, learningrate=learning_rate)

# load mnist training data CSV file into a list
training_data_file = open("mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for training
for e in range(epochs):
    # go through all the records in the training data set
    for record in training_data_list:
        # split the records by the commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        pass
    pass

# test the neural network

# load mnist test data CSV file into a list
test_data_file = open("mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# scorecard for how the neural network performs
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # print(correct_label, "correct label")
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # print(label, "network's answer")
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer
        scorecard.append(1)
    else:
        # network's answer doesn't match the correct answer
        scorecard.append(0)
        pass
    pass

# calculate score of neural network (fraction of correct answers)
scorecard_array = numpy.asarray(scorecard)
print("perforcmance = ", scorecard_array.sum() / scorecard_array.size)
