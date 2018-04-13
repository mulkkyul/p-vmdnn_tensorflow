import tensorflow as tf
import os
import time
import math
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
isdir = os.path.exists(flag.log_dir+'weights')
if not isdir:
    os.makedirs(flag.log_dir+'weights')
# if there's settings.ini, it will be overwritten!
copyfile('./settings.ini',flag.log_dir+'settings.ini')

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
            capacity= 3 * flag.batch_size
        )

        motor_batch, vision_batch, batch_idxd = tf.train.shuffle_batch(
            train_input_queue,
            batch_size=flag.batch_size,
            capacity= 3 * flag.batch_size,
            min_after_dequeue= 1*flag.batch_size,
            enqueue_many = False,
            allow_smaller_final_batch=True
        )

        #Model
        rnn_model = model.Model(motor_batch, vision_batch, batch_idxd, clratio, lr, flag)

    #For saving the network and logs
    saver = tf.train.Saver()  # saves the variables learned during training

    # open a text file to save loss during training
    f = open(os.path.join(flag.log_dir, "loss.txt"), 'w')


    with tf.Session(config=config) as sess:

        if flag.incremetalLearning:
            # If true, restore the trained network
            saver.restore(sess, flag.log_dir + flag.log_fn + "_epoch_%d" % (flag.restoreNetworkEpoch))
            print "Restoring the previously trained model... "
        else:
            # Otherwise, initialize the variables
            sess.run(tf.global_variables_initializer())

        # Some other settings before the training process
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tset = time.time()
        max_iter = int(math.ceil(flag.num_data/flag.batch_size))

        loss_minTH = flag.loss_minTH
        for epoch in xrange(flag.max_epoch):

            # Training the model
            for i in xrange(max_iter):
                sess.run(rnn_model.optimize, feed_dict={clratio: [flag.cl_ratio], lr:flag.lr})

            # Saving/Displaying the log files
            if (epoch + 1) % flag.print_epoch == 0:
                interval = time.time() - tset
                # For training loss
                loss_step = 0.0
                loss_step_m = 0.0
                loss_step_v = 0.0
                for i in xrange(max_iter):
                    tmp_loss, tmp_loss_m, tmp_loss_v = sess.run(rnn_model.cost, feed_dict={clratio: [flag.cl_ratio], lr:flag.lr})
                    loss_step += tmp_loss
                    loss_step_m += tmp_loss_m
                    loss_step_v += tmp_loss_v

                loss_step /= max_iter
                loss_step_m /= max_iter
                loss_step_v /= max_iter
                loss_step /= flag.batch_size
                loss_step_m /= flag.batch_size
                loss_step_v /= flag.batch_size

                print('\rStep %d. (TR) Loss: %.6f. P_LOSS: %.6f V_LOSS: %.6f Time(/%d epochs): %.4f, CL: %.2f LR: %s' % (
                    epoch + 1, loss_step, loss_step_m, loss_step_v, flag.print_epoch, interval, flag.cl_ratio, flag.lr))

                f.write("%d\t%.9f\t%.9f\t%.9f\n" % (epoch + 1, loss_step, loss_step_m, loss_step_v))
                tset = time.time()

            if (epoch + 1) % flag.saveNetwork_epoch == 0:
                ckpt_file = os.path.join(flag.log_dir, flag.log_fn + "_epoch_%d" % (epoch + 1))
                saver.save(sess, ckpt_file)
                ckpt_file = os.path.join(flag.log_dir, flag.log_fn + "_epoch_%d" % (epoch + 1) + '.index')
                dst_file = os.path.join(flag.log_dir + 'weights/', flag.log_fn + "_epoch_%d" % (epoch + 1) + '.index')
                copyfile(ckpt_file, dst_file)
                ckpt_file = os.path.join(flag.log_dir, flag.log_fn + "_epoch_%d" % (epoch + 1) + '.data-00000-of-00001')
                dst_file = os.path.join(flag.log_dir + 'weights/', flag.log_fn + "_epoch_%d" % (epoch + 1) + '.data-00000-of-00001')
                copyfile(ckpt_file, dst_file)
                ckpt_file = os.path.join(flag.log_dir, flag.log_fn + "_epoch_%d" % (epoch + 1) + '.meta')
                dst_file = os.path.join(flag.log_dir + 'weights/', flag.log_fn + "_epoch_%d" % (epoch + 1) + '.meta')
                copyfile(ckpt_file, dst_file)

        coord.request_stop()
        coord.join(threads)

        # Save the result from the last epoch (maximum epoch)
        ckpt_file = os.path.join(flag.log_dir, flag.log_fn + "_epoch_%d" % (epoch + 1))
        saver.save(sess, ckpt_file)
        ckpt_file = os.path.join(flag.log_dir, flag.log_fn + "_epoch_%d" % (epoch + 1) + '.index')
        dst_file = os.path.join(flag.log_dir + 'weights/', flag.log_fn + "_epoch_%d" % (epoch + 1) + '.index')
        copyfile(ckpt_file, dst_file)
        ckpt_file = os.path.join(flag.log_dir, flag.log_fn + "_epoch_%d" % (epoch + 1) + '.data-00000-of-00001')
        dst_file = os.path.join(flag.log_dir + 'weights/',
        flag.log_fn + "_epoch_%d" % (epoch + 1) + '.data-00000-of-00001')
        copyfile(ckpt_file, dst_file)
        ckpt_file = os.path.join(flag.log_dir, flag.log_fn + "_epoch_%d" % (epoch + 1) + '.meta')
        dst_file = os.path.join(flag.log_dir + 'weights/', flag.log_fn + "_epoch_%d" % (epoch + 1) + '.meta')
        copyfile(ckpt_file, dst_file)


        sess.close()


    f.close()

