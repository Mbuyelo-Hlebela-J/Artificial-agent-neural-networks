import math

A =[(1.0,2,8),
    (1,2,4),
    (1,2,2)]

D =[(1,2,8),
    (1,2,4),
    (1,2,2)]
B =[(5,8,10)]
e = 2.71828
def vector_add(A,B):
    if(len(A)==len(B)):
        res_vector = []
        for i in range(len(A)):
            row = []
            for j in range(len(A[i])):
                
                row.append(A[i][j]+B[i][j])
            res_vector.append(row)
        return res_vector
    print("The vectors are the same dimension")
    return 0
    
def dot_product(v1,v2):
    return sum(v1[i]*v2[i] for i in range(len(v1)))

def scalar_vector_mul(scalar,A):
    res_vector = []
    for i in range(len(A)):
        row = []
        for j in range(len(A[i])):
            row.append(A[i][j]*scalar)

        res_vector.appened(row)
    return res_vector


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)
def relu(x):
    return max(0.0,x)
def derivative_relu(x):
    return 1.0 if x>0 else 0.0
def tanh(x):
    return math.tanh(x)
def derivative_tanh(x):
    return 1 - pow(math.tanh(x),2)
def leaky_relu(x,alpha=0.01):
    return x if x > 0 else alpha * x
def derivative_leaky_relu(x,alpha=0.01):
    return 1.0 if x > 0 else alpha

def softmax(zs):
    exps =[math.exp(z) for z in zs]
    total = sum(exps)
    return [e/total for e in exps]


    


















    

