import numpy as np
import matplotlib.pyplot as plt

#1.     Load the training images
fileLocation = "F:/GUC/Semester 9/Machine Learning/Projects/Assignment1/Train/"
TrainingImages_array = []
for i in range(1,2401):
    image_location = fileLocation + str(i) + ".jpg"
    image = plt.imread(image_location)
    TrainingImages_array.append(image)
print("The number of training images is: \n",len(TrainingImages_array))

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
print("X Matrix: \n",X.shape)

#3.     Calculate X'Transpose
X_Transpose = np.transpose(X)
print("X Transpose Matrix: \n",X_Transpose.shape)

#4.     Calulate ((X'Transpose * X')^-1)*X'Transpose
X_Transpose_X = np.matmul(X_Transpose,X)
print("X Transpose x X: \n",X_Transpose_X.shape)

X_Transpose_X_inverse = np.linalg.pinv (X_Transpose_X)
print("(X Transpose x X)`1: \n",X_Transpose_X_inverse.shape) 

X_Transpose_X_inverse_X = np.matmul(X_Transpose_X_inverse,X_Transpose)
print("((X Transpose x X)`1)x X Transpose: \n",X_Transpose_X_inverse_X.shape)

#5.     Create the labels for the training dataset 10 classes --> t0 <----> t9
target_t0 = np.zeros(shape=(2400,1)).astype(int)
target_t1 = np.zeros(shape=(2400,1)).astype(int)
target_t2 = np.zeros(shape=(2400,1)).astype(int)
target_t3 = np.zeros(shape=(2400,1)).astype(int)
target_t4 = np.zeros(shape=(2400,1)).astype(int)
target_t5 = np.zeros(shape=(2400,1)).astype(int)
target_t6 = np.zeros(shape=(2400,1)).astype(int)
target_t7 = np.zeros(shape=(2400,1)).astype(int)
target_t8 = np.zeros(shape=(2400,1)).astype(int)
target_t9 = np.zeros(shape=(2400,1)).astype(int)

#6.     For example, T0 is the label for images with number zero on them, check in the label text file
#       where is the actual position of the zero images in all the 2400 images in the training set
#       Set the value of each Ti with (1) if the label contains the image with the same number as i

#7.     Open the label text file.
fileLocation = "F:/GUC/Semester 9/Machine Learning/Projects/Assignment1/Train/Training Labels.txt"
with open(fileLocation,'r') as textFile:
    trainingLabels = textFile.readlines()

labelsArray = []

for i in range(0,len(trainingLabels)):
    x = (int)(trainingLabels[i])
    labelsArray.append(x)

for i in range(0, len(labelsArray)):
    target = labelsArray[i]
    if(target == 0):
        target_t0[i] = 1
    else:
        target_t0[i] = -1

    if(target == 1):
        target_t1[i] = 1
    else:
        target_t1[i] = -1

    if(target == 2):
        target_t2[i] = 1
    else:
        target_t2[i] = -1

    if(target == 3):
        target_t3[i] = 1
    else:
        target_t3[i] = -1

    if(target == 4):
        target_t4[i] = 1
    else:
        target_t4[i] = -1

    if(target == 5):
        target_t5[i] = 1
    else:
        target_t5[i] = -1

    if(target == 6):
        target_t6[i] = 1
    else:
        target_t6[i] = -1

    if(target == 7):
        target_t7[i] = 1
    else:
        target_t7[i] = -1

    if(target == 8):
        target_t8[i] = 1
    else:
        target_t8[i] = -1
    
    if(target == 9):
        target_t9[i] = 1
    else:
        target_t9[i] = -1

#   Check that u calculated right, each t should have equal number of 240 ones and 2160 is zeros
# count1=0
# countNot1=0
# size = 2400
# for i in range (0,size):
#     if(target_t1[i] == 1):
#         count1 +=1
#     elif(target_t1[i] == -1):
#         countNot1 +=1

# print(count1)
# print(countNot1)

#8.     Finally, Calculate the Weight Vector for each of the 10 classes by multiplying the X shit with its target labels.

W0 = np.matmul(X_Transpose_X_inverse_X,target_t0)
W1 = np.matmul(X_Transpose_X_inverse_X,target_t1)    
W2 = np.matmul(X_Transpose_X_inverse_X,target_t2)    
W3 = np.matmul(X_Transpose_X_inverse_X,target_t3)    
W4 = np.matmul(X_Transpose_X_inverse_X,target_t4)    
W5 = np.matmul(X_Transpose_X_inverse_X,target_t5)    
W6 = np.matmul(X_Transpose_X_inverse_X,target_t6)    
W7 = np.matmul(X_Transpose_X_inverse_X,target_t7)    
W8 = np.matmul(X_Transpose_X_inverse_X,target_t8)    
W9 = np.matmul(X_Transpose_X_inverse_X,target_t9)

print("Weight Vector size: \n",W0.shape)

#9.     Get the equivalent Transpose for each of the Weight Vectors W'T
W0_Transpose = W0.transpose()
W1_Transpose = W1.transpose()
W2_Transpose = W2.transpose()
W3_Transpose = W3.transpose()
W4_Transpose = W4.transpose()
W5_Transpose = W5.transpose()
W6_Transpose = W6.transpose()
W7_Transpose = W7.transpose()
W8_Transpose = W8.transpose()
W9_Transpose = W9.transpose()

print("Weight Vector Transpose size: \n",W0_Transpose.shape)

#10.    Load the test images
fileLocation = "F:/GUC/Semester 9/Machine Learning/Projects/Assignment1/Test/"
TestImages_array = []
for i in range(1,201):
    image_location = fileLocation + str(i) + ".jpg"
    image = plt.imread(image_location)
    TestImages_array.append(image)
print("Number of test images: \n",len(TestImages_array))

#11.    Load the testing label text file
fileLocation = "F:/GUC/Semester 9/Machine Learning/Projects/Assignment1/Test/Test Labels.txt"
with open(fileLocation,'r') as textFile:
    testingLabels = textFile.readlines()

testlabelsArray = []

for i in range(0,len(testingLabels)):
    x = (int)(testingLabels[i])
    testlabelsArray.append(x)

#print(testlabelsArray)

#12.    Create a prediction array, and check the difference between the resuls of the prediction and the testLabelArray
#       To calculate the prediction array labels, u need to dot product all the weight vectors with the image coordinates
#       Then select the maximum value of this product
#       The index of that value indicates the classifier this image belongs to
#       Then compate ur results with the tested labels
predictionsArray = [0] * len(TestImages_array)
print("Number of predictions we should calculate:\n ",len(predictionsArray))

for i in range(0,len(TestImages_array)):
    image = TestImages_array[i]
    image_vector = image.flatten()
    # Add the bais to get the image ready for dot product with each value of the weight vectors
    image_vector = np.append(image_vector,1)
    #   Add all the results inside an array and select the maximum value to indicates the index of ur classifier
    classiferArray = []

    classifier0 = np.dot(W0_Transpose,image_vector)
    classifier1 = np.dot(W1_Transpose,image_vector)
    classifier2 = np.dot(W2_Transpose,image_vector)
    classifier3 = np.dot(W3_Transpose,image_vector)
    classifier4 = np.dot(W4_Transpose,image_vector)
    classifier5 = np.dot(W5_Transpose,image_vector)
    classifier6 = np.dot(W6_Transpose,image_vector)
    classifier7 = np.dot(W7_Transpose,image_vector)
    classifier8 = np.dot(W8_Transpose,image_vector)
    classifier9 = np.dot(W9_Transpose,image_vector)

    classiferArray.append(classifier0)
    classiferArray.append(classifier1)
    classiferArray.append(classifier2)
    classiferArray.append(classifier3)
    classiferArray.append(classifier4)
    classiferArray.append(classifier5)
    classiferArray.append(classifier6)
    classiferArray.append(classifier7)
    classiferArray.append(classifier8)
    classiferArray.append(classifier9)
   
    maxValue = np.amax(classiferArray)
    indexOfMaxValue = classiferArray.index(maxValue)
    predictionsArray[i] = indexOfMaxValue

#print(predictionsArray)

#13.    Calculate the accuracy
acc = 0
for i in range (0,200):
    if(testlabelsArray[i] == predictionsArray[i]):
        acc +=1

accuracy = (int)((acc/200) * 100)
print("The accuracy of our predicted calculations: \n",accuracy,"%")

#14.        Create a confusion Matrix
confusionMatrix = np.zeros((10, 10), dtype=int)
for i in range(len(testlabelsArray)):
    x = testlabelsArray[i]
    y = predictionsArray[i]
    confusionMatrix[x][y] += 1

print("The Confusion Matrix is: \n",confusionMatrix)

figure = plt.figure(figsize=(10,6))
plt.title("Confusion Matrix")
plt.imshow(confusionMatrix,cmap=plt.cm.Purples)
classesNames = ['0', '1','2','3','4','5','6','7','8','9']
tick_marks = np.arange(len(classesNames))
plt.xticks(tick_marks, classesNames, rotation=45)
plt.yticks(tick_marks, classesNames)
plt.ylabel('Actual Value')
plt.xlabel('Predicted Value')
plt.colorbar()
plt.tight_layout()
plt.show()

#15.        Save the image
figure.savefig("Confusion.jpg")


