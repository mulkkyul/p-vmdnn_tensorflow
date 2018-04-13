import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import ops
from shutil import copyfile
import pvmdnn_model as model
import utils

# ======================================================================================================================
# Read the model settings (options) & Dataset
# Argument: 0 for training, 1 for testing (error regression, entrainment)
flag, dbs = utils.readDataset(1)

# ======================================================================================================================
# If the log_dir doesn't exist, make a directory and copy the setting file
# Make a directory and copy the setting file
isdir = os.path.exists(flag.log_dir)
if not isdir:
    print 'Please check the log directory'
    assert False
isdir = os.path.exists(flag.log_dir+"entrainment")
if not isdir:
    os.makedirs(flag.log_dir+"entrainment")
copyfile('./settings.ini',flag.log_dir+ 'entrainment/' + 'settings.ini')

# Check the device (either CPU or GPU)
device_name = flag.device[0:4]
if device_name != '/cpu' and device_name != '/gpu':
    print 'The device should be either cpu or gpu'
    assert False
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)


# ======================================================================================================================
if __name__ == '__main__':
    with tf.device(flag.device):
        # Placeholder for clratio (Closed-loop ratio), lr (learning rate) and index (added in testing only(
        clratio = tf.placeholder(tf.float32, [1])
        lr = tf.placeholder(tf.float32, [])
        idxd = tf.placeholder(tf.int32, [None])


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

        wInit_vision = np.zeros((1, flag.out_size_vrow, flag.out_size_vcol))
        wInit_vision[0,:,:] = dbs.vision[0, 0, :, :]

        wInit_prop = np.zeros((1, flag.out_size_mdim, flag.out_size_smdim))
        wInit_prop[0, :, :] = dbs.prop[0, 0, :, :]

        # During the testing, initialize the initial states with the neutral value (0)
        wInit_p1 = np.zeros((flag.num_data, flag.p1_unit))
        wInit_p2 = np.zeros((flag.num_data, flag.p2_unit))
        wInit_p3 = np.zeros((flag.num_data, flag.p3_msize[0], flag.p3_msize[1], flag.p3_unit))
        
        wInit_v1 = np.zeros((flag.num_data, flag.v1_msize[0], flag.v1_msize[1], flag.v1_unit))
        wInit_v2 = np.zeros((flag.num_data, flag.v2_msize[0], flag.v2_msize[1], flag.v2_unit))
        wInit_v3 = np.zeros((flag.num_data, flag.v3_msize[0], flag.v3_msize[1], flag.v3_unit))

        # Setting the initial values for testing
        sess.run(rnn_model.set_wInitState,
                 feed_dict={rnn_model._windInit_p1_h: wInit_p1,
                            rnn_model._windInit_p2_h: wInit_p2,
                            rnn_model._windInit_p3_h: wInit_p3,
                            rnn_model._windInit_v1_h: wInit_v1,
                            rnn_model._windInit_v2_h: wInit_v2,
                            rnn_model._windInit_v3_h: wInit_v3})

        # Sensory Entrainment (without Prediction Error Minimization: this is just forward dynamics)
        pred_prop, c_p1, c_p2, c_p3, pred_vision, c_v1, c_v2, c_v3, input_t, init_state, iidx = sess.run(
            rnn_model.get_pred_m,
            feed_dict={clratio: [0.0], lr: flag.lr, idxd: [0],
                       rnn_model._windInput_prop: dbs.prop,
                       rnn_model._windTarget_prop: dbs.prop,
                       rnn_model._windInit_prop: wInit_prop,
                       rnn_model._windInput_vision: dbs.vision,
                       rnn_model._windTarget_vision: dbs.vision,
                       rnn_model._windInit_vision: wInit_vision})


        # Save Entrainment Results
        utils.save_entrainment_logs(input_t, pred_prop, c_p1, c_p2, c_p3, pred_vision, c_v1, c_v2, c_v3, flag)

        # Display Entrainment Results
        utils.display_entrainment_logs(input_t, pred_prop, c_p1, c_p2, c_p3, pred_vision, c_v1, c_v2, c_v3, flag)

        coord.request_stop()
        coord.join(threads)

        sess.close()


