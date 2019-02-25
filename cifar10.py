"""Builds the CIFAR-10 network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/.tensorflow/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.001  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
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
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader ops.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                                          data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def conv_2d(input_tensor, filters, name=None):
    with tf.variable_scope(name) as scope:
        print(scope.name, end="\t")
        print("\t{}".format(input_tensor.shape), end="\t")
        # kernel is filter, remember: `[filter_height, filter_width, in_channels, out_channels]`
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, input_tensor.shape[-1], filters],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [filters], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)
        print("\t{}".format(conv1.shape))
        return conv1


def pooling(input_tensor, name=None):
    with tf.variable_scope(name) as scope:
        print(scope.name, end="\t")
        print("\t{}".format(input_tensor.shape), end="\t")
        pool = tf.nn.max_pool(input_tensor, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='VALID', name=scope.name)
        print("\t{}".format(pool.shape))
        return pool


def fc(input_tensor, units, name=None, run_relu=True):
    with tf.variable_scope(name) as scope:
        input_size = input_tensor.shape[1].value

        if run_relu:
            weights = _variable_with_weight_decay('weights', shape=[input_size, units],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [units], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(input_tensor, weights, transpose_a=False) + biases, name=scope.name)
        else:
            weights = _variable_with_weight_decay('weights', [input_size, NUM_CLASSES],
                                                  stddev=1/input_size, wd=None)
            biases = _variable_on_cpu('biases', [units], tf.constant_initializer(0.0))
            local4 = tf.add(tf.matmul(input_tensor, weights, transpose_a=False), biases, name=scope.name)
        _activation_summary(local4)

        print("{}\t\t{}\t\t\t{}".format(scope.name, input_tensor.shape, local4.shape))
        return local4


def model_q1(images):
    return inference(images)


def model_q2(images):
    return inference(images, qn2=True)


def model_q3(images):
    return inference(images, qn3=True)


def inference(images, qn2=False, qn3=False):
    """Build the CIFAR-10 model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
      qn2: True iff we're running qn 2 model
      qn3: True iff we're running qn 3 model
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    print("Layer Name\t\tShape In\t\t\tShape Out\n-------------------------------------------------------------------")
    # level 1
    with tf.variable_scope("level1"):
        conv11 = conv_2d(images, 64, "conv1")
        conv12 = conv_2d(conv11, 64, "conv2")
        pool11 = pooling(conv12, "pool1")

    if not qn3:
        # level 2
        with tf.variable_scope("level2"):
            conv21 = conv_2d(pool11, 128, "conv1")
            conv22 = conv_2d(conv21, 128, "conv2")
            pool21 = pooling(conv22, "pool1")

    if not qn2:
        # level 3
        with tf.variable_scope("level3"):
            conv31 = conv_2d(pool11 if qn3 else pool21, 256, "conv1")
            conv32 = conv_2d(conv31, 256, "conv2")
            pool31 = pooling(conv32, "pool1")

    # level 4
    with tf.variable_scope("level4"):
        conv41 = conv_2d(pool21 if qn2 else pool31, 512, "conv1")
        conv42 = conv_2d(conv41, 512, "conv2")

    # level 5
    with tf.variable_scope("level5"):
        # reshape
        with tf.variable_scope("reshape") as scope:
            print(scope.name, end="\t")
            print("\t{}".format(conv42.shape), end="\t")
            conv_shape = conv42.shape
            reshape = tf.reshape(conv42, [conv_shape[0], conv_shape[1]*conv_shape[2]*conv_shape[3]], name=scope.name)
            print("\t{}".format(reshape.shape))

        fc1 = fc(reshape, 200, "fc-200")
        fc2 = fc(fc1, 100, "fc-100")
        logits = fc(fc2, NUM_CLASSES, "logits", run_relu=False)

        return logits


def accuracy(logits, labels):
    # labels is not one-hot encoded. logits' shape is (batch_size, n_classes)
    # while labels' is (batch_size,)
    acc, acc_op = tf.metrics.accuracy(predictions=tf.argmax(logits, 1), labels=labels)
    return acc_op


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    print("loss - Shapes: logits: {}. labels: {}.".format(logits.shape, labels.shape))
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name, l)
        tf.summary.scalar(l.op.name + ' (m_avg)', loss_averages.average(l))

    return loss_averages_op


def train(total_loss, model_accuracy, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      model_accuracy: Model accuracy
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Accuracy
    tf.summary.scalar("accuracy", model_accuracy)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
