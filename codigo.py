import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn import svm
from sklearn.svm import SVC

test_dados = []
#Como nosso dataset contem string na coluna Y (classes) faremos uma substituicao (encode) da string por um vetor
def label_encode(label):
	val = []
	if label == "0.0":
		val = [1,0,0]
		
	elif label == "0.5":
		val = [0,1,0]
		
	elif label == "1.0":
		val = [0,0,1]
		
	return val

def label_encode1(label):
	
	if label == "0.0":
		val = 0
		
	elif label == "0.5":
		val = 1
		
	elif label == "1.0":
		val = 2
		
	return val

def data_encode(file):
	X = []
	Y = []
	Z = []
	train_file = open(file, 'r')
	for line in train_file.read().strip().split('\n'):
		line = line.split(',')
		X.append([line[0],line[1],line[2],line[3],line[4],line[5],line[6]])
		Y.append(label_encode(line[7]))
		Z.append(label_encode1(line[7]))
	return X,Y,Z

file = "train_nml.txt"
train_X, train_Y, testt = data_encode(file)



#parametros da rede
learning_rate = 0.01
training_epochs = 5000
display_steps = 100

n_input = 7 #quantos valores de entrada?
n_hidden = 20 #quantos neurônios na camada oculta?
n_output = 3 #quantos neuronios na camada de saida?

#a partir daqui construimos o modelo
X = tf.placeholder("float",[None,n_input])
Y = tf.placeholder("float",[None,n_output])

weights = {
	"hidden": tf.Variable(tf.random_normal([n_input,n_hidden])),
	"output": tf.Variable(tf.random_normal([n_hidden,n_output])),
}

bias = {
	"hidden": tf.Variable(tf.random_normal([n_hidden])),
	"output": tf.Variable(tf.random_normal([n_output])),
}

def model(X, weights, bias):
	layer1 = tf.add(tf.matmul(X, weights["hidden"]),bias["hidden"])
	layer1 = tf.nn.relu(layer1)

	output_layer = tf.matmul(layer1,weights["output"]) + bias["output"]
	
	return output_layer



test_X, test_Y, testy = data_encode("teste_nml.txt") #dataset de validacao


pred = model(X,weights,bias)

clf = svm.SVC(kernel='poly', C=4.0, gamma=2.0)

clf.fit(train_X,testt)


print(classification_report(testy, clf.predict(test_X)))
print(confusion_matrix(testy, clf.predict(test_X)))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizador = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epochs in range(training_epochs):
		_, c= sess.run([optimizador,cost],feed_dict = {X: train_X, Y: train_Y})
		if(epochs + 1) % display_steps == 0:
			print("Epoch:",epochs+1,"Cost:", c)
	print("Optimization Finished")

	test_result = sess.run(pred,feed_dict = {X: train_X})
	correct_prediction = tf.equal(tf.argmax(test_result,1),tf.argmax(train_Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))


	test_resultY = sess.run(pred,feed_dict = {X: test_X})
	test_pred = test_resultY.argmax(axis=1)
	
	print(classification_report(testy, test_pred))
	print(confusion_matrix(testy, test_pred))
	