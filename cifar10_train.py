"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/.tensorflow/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 60000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train(model_fn, train_folder, qn_id):
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model_fn(images)

        # Calculate loss.
        loss = cifar10.loss(logits, labels)

        # Calculate accuracy
        model_accuracy = cifar10.accuracy(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        global_step = tf.train.get_or_create_global_step()
        train_op = cifar10.train(loss, model_accuracy, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._start_time = time.time()

            def after_create_session(self, session, coord):
                self._step = session.run(global_step)

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs([loss, model_accuracy])  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results[0]
                    acc_value = run_values.results[1]
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s - %s: step %d, loss = %.2f, acc = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (qn_id, datetime.now(), self._step, loss_value, acc_value,
                                        examples_per_sec, sec_per_batch))

        class _StopAtHook(tf.train.SessionRunHook):
            def __init__(self, last_step):
                self._last_step = last_step

            def after_create_session(self, session, coord):
                self._step = session.run(global_step)

            def before_run(self, run_context):  # pylint: disable=unused-argument
                self._step += 1
                return tf.train.SessionRunArgs(global_step)

            def after_run(self, run_context, run_values):
                if self._step >= self._last_step:
                    run_context.request_stop()
        # class _StopAtHook(tf.train.StopAtStepHook):
        #     def __init__(self, last_step):
        #         super().__init__(last_step=last_step)
        #
        #     def begin(self):
        #         self._global_step_tensor = global_step
        #
        #     def before_run(self, run_context):  # pylint: disable=unused-argument
        #         return tf.train.SessionRunArgs(global_step)
        #
        #     def after_run(self, run_context, run_values):
        #         gs = run_values.results + 1
        #         print("\tgs = {}/{}".format(gs, self._last_step))
        #         if gs >= self._last_step:
        #             # Check latest global step to ensure that the targeted last step is
        #             # reached. global_step read tensor is the value of global step
        #             # before running the operation. We're not sure whether current session.run
        #             # incremented the global_step or not. Here we're checking it.
        #
        #             step = run_context.session.run(self._global_step_tensor)
        #             print("\t\tstep: {}. gs = {}/{}".format(step, gs, self._last_step))
        #             if step >= self._last_step:
        #                 run_context.request_stop()

        saver = tf.train.Saver()
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=train_folder,
                hooks=[_StopAtHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss), _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            latest_checkpoint_path = tf.train.latest_checkpoint(train_folder)
            if latest_checkpoint_path is not None:
                # Restore from checkpoint
                print("Restoring checkpoint from %s" % latest_checkpoint_path)
                saver.restore(mon_sess, latest_checkpoint_path)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def run_training(model_fn, qn_id):
    cifar10.maybe_download_and_extract()
    train_folder = FLAGS.train_dir + "_" + qn_id
    # if tf.gfile.Exists(train_folder):
    #     tf.gfile.DeleteRecursively(train_folder)
    # tf.gfile.MakeDirs(train_folder)
    train(model_fn, train_folder, qn_id)
    print("Done running training for " + qn_id + "\n===================================\n")
    time.sleep(15)


def main(argv=None):  # pylint: disable=unused-argument
    run_training(cifar10.model_q1, "q1")
    run_training(cifar10.model_q2, "q2")
    run_training(cifar10.model_q3, "q3")


if __name__ == '__main__':
    tf.app.run()
