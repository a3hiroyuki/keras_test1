
nb_classes = 10

import tensorflow as tf
import os
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np

output_dir = "/home/abe/frozen/"
output_graph_name ="keras2tf.pb"

def load_graph(model_file):
    #graph = tf.Graph()
    graph = tf.get_default_graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
        #return_elements  = tf.import_graph_def(graph_def,  name="", return_elements=['input_1:0', 'output0:0'])
#         X = return_elements[0]
#         Y = return_elements[1]
    return graph


tfmodel = load_graph(os.path.join(output_dir, output_graph_name))

# print operations in the model, in tf.Operation format

for op in tfmodel.get_operations():
    print(op.name)


# inLayer = tfmodel.get_operation_by_name('import/input_1')
# learnPhase = tfmodel.get_operation_by_name('import/dropout_1/keras_learning_phase')
# outLayer = tfmodel.get_operation_by_name('import/output0')

# use the model with test data to predict the label

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

batch_size =128

X = tf.placeholder("float", [batch_size, 32, 32, 3])
Y = tf.placeholder("float", [batch_size, 10])
learning_rate = tf.placeholder("float", [])
#
# cross_entropy = -tf.reduce_sum(Y*tf.log(tfmodel))
# opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
# train_op = opt.minimize(cross_entropy)

input_layer = "input"
output_layer = "output"

input_name = "import/input_1"
learning_name = 'import/batch_normalization_1/keras_learning_phase'
softmax_name = "import/dense_1/Softmax"
output_name = "import/output0"
input_operation = tfmodel.get_operation_by_name(input_name)
softmax_operation = tfmodel.get_operation_by_name(softmax_name)
learning_operation = tfmodel.get_operation_by_name(learning_name)
output_operation = tfmodel.get_operation_by_name(output_name)

print (input_operation.outputs[0])
print(output_operation.outputs[0])
print(softmax_operation.outputs[0])
print(learning_operation.outputs[0])

learning_rate = tf.placeholder("float", [])

#cross_entropy = -tf.reduce_sum(output_operation.outputs[0] * tf.log(softmax_operation.outputs[0]))
#opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
optimizer = tf.train.AdamOptimizer()
#print (cross_entropy)
#train_op = optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(softmax_operation.outputs[0], 1), tf.argmax(output_operation.outputs[0], 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session(graph=tfmodel) as sess:
#     results = sess.run(outLayer.outputs[0],
#                        {inLayer.outputs[0]: X_test,
#                         learnPhase.outputs[0]: 0})
#     for j in range (2):
#         for i in range (0, 20000, batch_size):
#             feed_dict = {
#                     input_operation.outputs[0]: X_train[i:i + batch_size],
#                     output_operation.outputs[0]: Y_train[i:i + batch_size]}
#             sess.run(softmax_operation.outputs[0], feed_dict=feed_dict)
#             if i % 512 == 0:
#                 print ("training on image #%d" % i)
    for i in range (0, 10000, batch_size):
        if i + batch_size < 10000:
            acc = sess.run([accuracy],
            feed_dict={
                input_operation.outputs[0]: X_test[i:i+batch_size],
                output_operation.outputs[0]: Y_test[i:i+batch_size]
            })
        #accuracy_summary = tf.scalar_summary("accuracy", accuracy)
        print (acc)

    sess.close()



