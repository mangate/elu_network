__author__ = 'mangate'

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from six.moves import cPickle as pickle
from open_cifar100_data import get_data


NUM_CLASSES = 100
train_images, train_labels, test_images,test_labels = get_data()

def save_data_to_file(filename,data):
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print('Data saved to',filename)
    except Exception as e:
        print('Unable to save data to', filename, ':', e)

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    data = cPickle.load(fo)
    fo.close()
    return data

def _variable_on_cpu(name, shape, initializer=tf.constant_initializer(0.1)):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev=0.05, wd=0.0005):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

#Accuracy measurment for the results
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

graph = tf.Graph()
with graph.as_default():
    x  = tf.placeholder(tf.float32, [None, 32, 32, 3],name="x-input")
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES],name="y-input")

    #Dropout rates
    stack1_prob_input = tf.placeholder(tf.float32)
    stack2_prob_input = tf.placeholder(tf.float32)
    stack3_prob_input = tf.placeholder(tf.float32)
    stack4_prob_input = tf.placeholder(tf.float32)
    stack5_prob_input = tf.placeholder(tf.float32)
    stack6_prob_input = tf.placeholder(tf.float32)
    stack7_prob_input = tf.placeholder(tf.float32)

    w1_stack1 = _variable_with_weight_decay('w1_stack1',shape=[5,5,3,192])
    b1_stack1 = _variable_on_cpu('b1_stack1',shape=[192])

    w1_stack2 = _variable_with_weight_decay('w1_stack2',shape=[1,1,192,192])
    b1_stack2 = _variable_on_cpu('b1_stack2',shape=[192])
    w2_stack2 = _variable_with_weight_decay('w2_stack2',shape=[3,3,192,240])
    b2_stack2 = _variable_on_cpu('b2_stack2',shape=[240])

    w1_stack3 = _variable_with_weight_decay('w1_stack3',shape=[1,1,240,240])
    b1_stack3 = _variable_on_cpu('b1_stack3',shape=[240])
    w2_stack3 = _variable_with_weight_decay('w2_stack3',shape=[2,2,240,260])
    b2_stack3 = _variable_on_cpu('b2_stack3',shape=[260])

    w1_stack4 = _variable_with_weight_decay('w1_stack4',shape=[1,1,260,260])
    b1_stack4 = _variable_on_cpu('b1_stack4',shape=[260], initializer=tf.constant_initializer(0.1))
    w2_stack4 = _variable_with_weight_decay('w2_stack4',shape=[2,2,260,280])
    b2_stack4 = _variable_on_cpu('b2_stack4',shape=[280])

    w1_stack5 = _variable_with_weight_decay('w1_stack5',shape=[1,1,280,280])
    b1_stack5 = _variable_on_cpu('b1_stack5',shape=[280])
    w2_stack5 = _variable_with_weight_decay('w2_stack5',shape=[2,2,280,300])
    b2_stack5 = _variable_on_cpu('b2_stack5',shape=[300])

    w1_stack6 = _variable_with_weight_decay('w1_stack6',shape=[1,1,300,300])
    b1_stack6 = _variable_on_cpu('b1_stack6',shape=[300])

    w1_stack7 = _variable_with_weight_decay('w1_stack7',shape=[1,1,300,100])
    b1_stack7 = _variable_on_cpu('b1_stack7',shape=[100])

    def model(input_data):
        #Stack 1
        #========

        conv1_stack1 = tf.nn.conv2d(input_data, w1_stack1, [1, 1, 1, 1], padding='SAME')
        bias1 = tf.nn.bias_add(conv1_stack1, b1_stack1)
        stack1 = tf.nn.elu(bias1)
        stack1_dropped=tf.nn.dropout(stack1,stack1_prob_input)

        #pooling
        pool1 = tf.nn.max_pool(stack1_dropped, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')

        #Stack2
        #========

        conv1_stack2 = tf.nn.conv2d(pool1, w1_stack2, [1, 1, 1, 1], padding='SAME')
        bias1_stack2 = tf.nn.bias_add(conv1_stack2, b1_stack2)
        stack2_1 = tf.nn.elu(bias1_stack2)

        conv2_stack2 = tf.nn.conv2d(stack2_1, w2_stack2, [1, 1, 1, 1], padding='SAME')
        bias2_stack2 = tf.nn.bias_add(conv2_stack2, b2_stack2)
        stack2_2 = tf.nn.elu(bias2_stack2)
        stack2_dropped=tf.nn.dropout(stack2_2,stack2_prob_input)

        #pooling
        pool2 = tf.nn.max_pool(stack2_dropped, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')


        #Stack3
        #========

        conv1_stack3 = tf.nn.conv2d(pool2, w1_stack3, [1, 1, 1, 1], padding='SAME')
        bias1_stack3 = tf.nn.bias_add(conv1_stack3, b1_stack3)
        stack3_1 = tf.nn.elu(bias1_stack3)


        conv2_stack3 = tf.nn.conv2d(stack3_1, w2_stack3, [1, 1, 1, 1], padding='SAME')
        bias2_stack3 = tf.nn.bias_add(conv2_stack3, b2_stack3)
        stack3_2 = tf.nn.elu(bias2_stack3)
        stack3_dropped=tf.nn.dropout(stack3_2,stack3_prob_input)

        #pooling
        pool3 = tf.nn.max_pool(stack3_dropped, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')



        #Stack4
        #========

        conv1_stack4 = tf.nn.conv2d(pool3, w1_stack4, [1, 1, 1, 1], padding='SAME')
        bias1_stack4 = tf.nn.bias_add(conv1_stack4, b1_stack4)
        stack4_1 = tf.nn.elu(bias1_stack4)

        conv2_stack4 = tf.nn.conv2d(stack4_1, w2_stack4, [1, 1, 1, 1], padding='SAME')
        bias2_stack4 = tf.nn.bias_add(conv2_stack4, b2_stack4)
        stack4_2 = tf.nn.elu(bias2_stack4)
        stack4_dropped=tf.nn.dropout(stack4_2,stack4_prob_input)

        #pooling
        pool4 = tf.nn.max_pool(stack4_dropped, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')

        #Stack5
        #========

        conv1_stack5 = tf.nn.conv2d(pool4, w1_stack5, [1, 1, 1, 1], padding='SAME')
        bias1_stack5 = tf.nn.bias_add(conv1_stack5, b1_stack5)
        stack5_1 = tf.nn.elu(bias1_stack5)

        conv2_stack5 = tf.nn.conv2d(stack5_1, w2_stack5, [1, 1, 1, 1], padding='SAME')
        bias2_stack5 = tf.nn.bias_add(conv2_stack5, b2_stack5)
        stack5_2 = tf.nn.elu(bias2_stack5)
        stack5_dropped=tf.nn.dropout(stack5_2,stack5_prob_input)

        #pooling
        pool5 = tf.nn.max_pool(stack5_dropped, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')

        #Stack6
        #========

        conv1_stack6 = tf.nn.conv2d(pool5, w1_stack6, [1, 1, 1, 1], padding='SAME')
        bias1_stack6 = tf.nn.bias_add(conv1_stack6, b1_stack6)
        stack6_1 = tf.nn.elu(bias1_stack6)

        stack6_dropped=tf.nn.dropout(stack6_1,stack6_prob_input)

        #pooling
        #pool6 = tf.nn.max_pool(stack6_dropped, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                     padding='SAME', name='pool1')

        #Stack7
        #========

        conv1_stack7 = tf.nn.conv2d(stack6_dropped, w1_stack7, [1, 1, 1, 1], padding='SAME')
        bias1_stack7 = tf.nn.bias_add(conv1_stack7, b1_stack7)
        stack7_1 = tf.nn.elu(bias1_stack7)

        stack7_dropped=tf.nn.dropout(stack7_1,stack7_prob_input)

        #pooling
        #pool7 = tf.nn.max_pool(stack7_dropped, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        #                     padding='SAME', name='pool1')

        #Softmax layer
        y_conv_reshaped=tf.reshape(stack7_dropped,(-1,NUM_CLASSES))

        return y_conv_reshaped

    #Loss
    #======
    #labels = tf.cast(y_, tf.int64)
    results,_ = model(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    results, y_, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    #Total loss
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    #optimizer
    #==========
    tf_learning_rate = tf.placeholder(tf.float32)
    optimizer = tf.train.MomentumOptimizer(learning_rate=tf_learning_rate, momentum=0.9).minimize(total_loss)
    #Predictions
    tf_prediction = tf.nn.softmax(results)

    saver = tf.train.Saver()

#Training
#==========
with tf.Session(graph=graph) as sess:
    init=tf.initialize_all_variables()
    sess.run(init)
    learning_rate = 0.01
    batch_size = 100
    step_per_epoch = int(len(train_labels)/batch_size)
    for step in range(165000):
      #Shuffeling the data on each epoch
      if step % step_per_epoch == 0:
             shuffle_indices = np.random.permutation(np.arange(len(train_images)))
             train_images = train_images[shuffle_indices]
             train_labels = train_labels[shuffle_indices]
             indices_list = create_indices(train_labels,NUM_CLASSES)
      if step == 35000:
          learning_rate = 0.005
      if step == 85000:
          learning_rate = 0.0005
      if step == 135000:
          learning_rate = 0.00005
      #Creating batch data
      offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
      batch_data = train_images[offset:(offset + batch_size), :, :, :]
      batch_labels = train_labels[offset:(offset + batch_size), :]
      #Creating the pairwise data
      x1_pair_in,x2_pair_in,y_pair_in = create_pairwise_batch(train_images,train_labels,indices_list,batch_size)
      #Training
      feed_dict = {x: batch_data, y_ : batch_labels, tf_learning_rate: learning_rate,
                   x1_pair: x1_pair_in, x2_pair: x2_pair_in, y_pair: y_pair_in,
                   stack1_prob_input: 1.0, stack2_prob_input: 0.9, stack3_prob_input: 0.8,
                   stack4_prob_input: 0.7, stack5_prob_input: 0.6, stack6_prob_input: 0.5, stack7_prob_input: 1.0}
      _, loss_out, predictions = sess.run([optimizer, total_loss, tf_prediction], feed_dict=feed_dict)
      #Accuracy
      if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, loss_out))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          accuracy_final = 0.0
          test_predictions = np.ndarray(shape=(len(test_labels),NUM_CLASSES),dtype=np.float32)
          for i in range(20):
                offset = i*500
                feed_dict = {x: test_images[offset:(offset+500)], y_ : test_labels[offset:(offset+500)], stack1_prob_input: 1.0,
                                 stack2_prob_input: 1.0, stack3_prob_input: 1.0, stack4_prob_input: 1.0, stack5_prob_input: 1.0,
                                 stack6_prob_input: 1.0, stack7_prob_input: 1.0}
                test_predictions[offset:(offset+500)] = sess.run(tf_prediction, feed_dict = feed_dict)
                accuracy_final+=accuracy(test_predictions[offset:(offset+500)], test_labels[offset:(offset+500)])
          print('elu network 80000 steps')
          print('Test accuracy is %.1f%%' %(accuracy_final/20))

        #Save the model
        saver.save(sess, 'ckpt/elu_model.ckpt', global_step=80000)
