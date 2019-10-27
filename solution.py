import numpy as np
import matplotlib.pyplot as plt

#1.     Load the training images
fileLocation = "F:/GUC/Semester 9/Machine Learning/Projects/Train/"
TrainingImages_array = []
for i in range(1,2401):
    image_location = fileLocation + str(i) + ".jpg"
    image = plt.imread(image_location)
    TrainingImages_array.append(image)
print(len(TrainingImages_array))

#2.     Create the X'
#2.1 Here we create a matrix holds the images vectors appeneded with bais 1 so the size is 784 + 1
X = np.zeros(shape=(1,785)).astype(int)

#2.2 Loop in the image arrays to create the image vectors
for i in range(0,len(TrainingImages_array)):
    image = TrainingImages_array[i]
    image_vector = image.flatten()
    vector_appends_1 = np.append(image_vector,1)
    X = np.vstack((X,vector_appends_1))
X = np.delete(X, (0), axis=0)
print(X.shape)

#3.     Calculate X'Transpose
X_Transpose = np.transpose(X)
print(X_Transpose.shape)

#4.     Calulate ((X'Transpose * X')^-1)*X'Transpose
X_Transpose_X = np.matmul(X_Transpose,X)
print(X_Transpose_X.shape)

X_Transpose_X_inverse = np.linalg.pinv (X_Transpose_X)
print(X_Transpose_X_inverse.shape) 

X_Transpose_X_inverse_X = np.matmul(X_Transpose_X_inverse,X_Transpose)
print(X_Transpose_X_inverse_X.shape)