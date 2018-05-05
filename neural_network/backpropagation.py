
## Example from Siraj Raval https://www.youtube.com/watch?v=q555kfIFUCM&t=260s

import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
Y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
weights0 = 2*np.random.random((3,4)) - 1
weights1 = 2*np.random.random((4,1)) - 1

for j in range(60000):

	# Feed forward through layers 0, 1, and 2
    layer0 = X
    layer1 = nonlin(np.dot(layer0,weights0))
    layer2 = nonlin(np.dot(layer1,weights1))

    # how much did we miss the target value?
    layer2_error = Y - layer2

    if (j % 10000) == 0:
        print(str(np.mean(np.abs(layer2_error))))
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    layer2_delta = layer2_error*nonlin(layer2,deriv=True)

    # how much did each k1 value contribute to the k2 error (according to the weights)?
    layer1_error = layer2_delta.dot(weights1.T)
    
    # in what direction is the target k1?
    # were we really sure? if so, don't change too much.
    layer1_delta = layer1_error * nonlin(layer1,deriv=True)

    weights1 += layer1.T.dot(layer2_delta)
    weights0 += layer0.T.dot(layer1_delta)


input_data = np.array([ [0,0,0], [1,1,1], [1,0,1], [1, 0, 0] ])
expected_output_data = np.array([[0, 1, 1, 0]])

test_layer0 = input_data
test_layer1 = nonlin(np.dot(test_layer0, weights0))
test_layer2 = nonlin(np.dot(test_layer1, weights1))

print(test_layer2)