import numpy as np
d = 3
# Define basis functions Z=ReLU(W*x+B)
W = [] # Basis function weights
B = [] # Basis function bias

lb = [-5] * d
ub = [15] * d

# Add a constant basis function independent on the variable x, giving the model an offset
W.append([0]*d)
B.append([1])

# Add basis functions dependent on one variable
for k in range(d): 
    for i in range(lb[k],ub[k]+1):
        if i == lb[k]:
            temp = [0]*d
            temp[k] = 1
            W.append(np.copy(temp))
            B.append([-i])
        elif i == ub[k]:
            temp = [0]*d
            temp[k] = -1
            W.append(np.copy(temp))
            B.append([i])
        else:
            temp = [0]*d
            temp[k] = -1
            W.append(np.copy(temp))
            B.append([i])
            temp = [0]*d
            temp[k] = 1
            W.append(np.copy(temp))
            B.append([-i])
            
# Add basis functions dependent on two subsequent variables
for k in range(1,d): 
    for i in range(lb[k]-ub[k-1],ub[k]-lb[k-1]+1):
        if i == lb[k]-ub[k-1]:
            temp = [0]*d
            temp[k] = 1
            temp[k-1] = -1
            W.append(np.copy(temp))
            B.append([-i])
        elif i == ub[k]-lb[k-1]:
            temp = [0]*d
            temp[k] = -1
            temp[k-1] = 1
            W.append(np.copy(temp))
            B.append([i])
        else:
            temp = [0]*d
            temp[k] = -1
            temp[k-1] = 1
            W.append(np.copy(temp))
            B.append([i])
            temp = [0]*d
            temp[k] = 1
            temp[k-1] = -1
            W.append(np.copy(temp))
            B.append([-i])
        
W = np.asarray(W)	
B = np.asarray(B)	