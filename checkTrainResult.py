import tensorflow as tf
import os
from tensorflow.python.framework import ops
from shutil import copyfile
import pvmdnn_model as model
import utils

# ======================================================================================================================
# Read the model settings (options) & Dataset
# Argument: 0 for training, 1 for testing (error regression, entrainment)
flag, dbs = utils.readDataset(0)

# ======================================================================================================================
# If the log_dir doesn't exist, make a directory and copy the setting file
isdir = os.path.exists(flag.log_dir)
if not isdir:
    os.makedirs(flag.log_dir)
isdir = os.path.exists(flag.log_dir + 'weights')
if not isdir:
    os.makedirs(flag.log_dir + 'weights')
# if there's settings.ini, it will be overwritten!
copyfile('./settings.ini', flag.log_dir + 'settings.ini')

# Check the device (either CPU or GPU)
device_name = flag.device[0:4]
if device_name != '/cpu' and device_name != '/gpu':
    print 'The device should be either cpu or gpu'
    assert False
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

# ======================================================================================================================
if __name__ == '__main__':
    with tf.device(flag.device):
        # Placeholder for clratio (Closed-loop ratio) and lr (learning rate)
        clratio = tf.placeholder(tf.float32, [1])
        lr = tf.placeholder(tf.float32, [])

        # Preparing the dataset
        all_prop = ops.convert_to_tensor(dbs.prop, dtype=tf.float32)
        all_vision = ops.convert_to_tensor(dbs.vision, dtype=tf.float32)
        all_idxd = ops.convert_to_tensor(dbs.idxd, dtype=tf.int32)

        train_input_queue = tf.train.slice_input_producer(
            [all_prop, all_vision, all_idxd],
            shuffle=False,
            capacity=3 * flag.batch_size
        )

        motor_batch, vision_batch, batch_idxd = tf.train.shuffle_batch(
            train_input_queue,
            batch_size=flag.batch_size,
            capacity=3 * flag.batch_size,
            min_after_dequeue=1 * flag.batch_size,
            enqueue_many=False,
            allow_smaller_final_batch=True
        )

        # Model
        rnn_model = model.Model(motor_batch, vision_batch, batch_idxd, clratio, lr, flag)

    # For loading the trained network
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        # Restore the trained network
        saver.restore(sess, flag.log_dir + flag.log_fn + "_epoch_%d" % (flag.restoreNetworkEpoch))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Save the results
        utils.save_results(sess, rnn_model, flag)
        # Plot the proprioceptive and visual target/predictions
        utils.plot_outputs(sess, rnn_model, flag)

        coord.request_stop()
        coord.join(threads)
        sess.close()
