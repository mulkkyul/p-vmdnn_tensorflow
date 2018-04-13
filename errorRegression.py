import tensorflow as tf
import numpy as np
import os
import time
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
isdir = os.path.exists(flag.log_dir + "errorRegression")
if not isdir:
    os.makedirs(flag.log_dir + "errorRegression")
copyfile('./settings.ini', flag.log_dir + 'errorRegression/' + 'settings.ini')

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

    # Logging the loss during the error regression
    f = open(os.path.join(flag.log_dir + "errorRegression", "loss_errorRegression.txt"), 'w')
    if flag.ers_save_verbose:
        f_pe = open(os.path.join(flag.log_dir + "errorRegression", "pe_errorRegression.txt"), 'w')
    with tf.Session(config=config) as sess:

        # Restore the trained network
        saver.restore(sess, flag.log_dir + flag.log_fn + "_epoch_%d" % (flag.restoreNetworkEpoch))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        tset = time.time()
        max_iter = flag.numIterations

        log_output_vision = np.zeros((1, flag.length, flag.out_size_vrow, flag.out_size_vcol))
        log_target_vision = np.zeros((1, flag.length, flag.out_size_vrow, flag.out_size_vcol))
        wInit_vision = np.zeros((1, flag.out_size_vrow, flag.out_size_vcol))
        wInit_vision[0,:,:] = dbs.vision[0, 0, :, :]

        log_output_prop = np.zeros((1, flag.length, flag.out_size_mdim, flag.out_size_smdim))
        log_target_prop = np.zeros((1, flag.length, flag.out_size_mdim, flag.out_size_smdim))
        wInit_prop = np.zeros((1, flag.out_size_mdim, flag.out_size_smdim))
        wInit_prop[0, :, :] = dbs.prop[0, 0, :, :]


        wInit_p1 = np.zeros((flag.num_data, flag.p1_unit))
        wInit_p2 = np.zeros((flag.num_data, flag.p2_unit))
        wInit_p3 = np.zeros((flag.num_data, flag.p3_msize[0], flag.p3_msize[1], flag.p3_unit))

        wInit_v1 = np.zeros((flag.num_data, flag.v1_msize[0], flag.v1_msize[1], flag.v1_unit))
        wInit_v2 = np.zeros((flag.num_data, flag.v2_msize[0], flag.v2_msize[1], flag.v2_unit))
        wInit_v3 = np.zeros((flag.num_data, flag.v3_msize[0], flag.v3_msize[1], flag.v3_unit))
        

        # Before starting the ERS, initialize the initial states within the window with the neutral value (0)
        sess.run(rnn_model.set_wInitState,
                 feed_dict={rnn_model._windInit_p1_h: wInit_p1,
                            rnn_model._windInit_p2_h: wInit_p2,
                            rnn_model._windInit_p3_h: wInit_p3,
                            rnn_model._windInit_v1_h: wInit_v1,
                            rnn_model._windInit_v2_h: wInit_v2,
                            rnn_model._windInit_v3_h: wInit_v3})

        length = flag.length

        print_epoch = max_iter
        cl_schedule = 1.0

        # --------------------------------------------------------- #
        log_iteration = np.zeros((flag.length))
        # --------------------------------------------------------- #

        #for t in xrange(1, length):
        for t in xrange(1, 50):
            if (t == 1):
                input_vision = dbs.vision[:, 0, :, :]
                input_vision = np.reshape(input_vision,(1, 1, flag.out_size_vrow, flag.out_size_vcol))
                target_vision = dbs.vision[:, 1, :, :]
                target_vision = np.reshape(target_vision, (1, 1, flag.out_size_vrow, flag.out_size_vcol))

                input_prop = dbs.prop[:, 0, :, :]
                input_prop = np.reshape(input_prop, (1, 1, flag.out_size_mdim, flag.out_size_smdim))
                target_prop = dbs.prop[:, 1, :, :]
                target_prop = np.reshape(target_prop, (1, 1, flag.out_size_mdim, flag.out_size_smdim))
            elif (t <= flag.windowLength):
                input_vision = dbs.vision[:, 0:t - 1, :, :]
                target_vision = dbs.vision[:, 1:t, :, :]

                input_prop = dbs.prop[:, 0:t - 1, :, :]
                target_prop = dbs.prop[:, 1:t, :, :]
            else:
                input_vision = dbs.vision[:, t - flag.windowLength:t - 1, :, :]
                target_vision = dbs.vision[:, t - flag.windowLength + 1:t, :, :]

                input_prop = dbs.prop[:, t - flag.windowLength:t - 1, :, :]
                target_prop = dbs.prop[:, t - flag.windowLength + 1:t, :, :]

            log_iteration[t-1] = max_iter

            for i in xrange(max_iter):
                sess.run(rnn_model.optimize_testing,
                         feed_dict={clratio: [cl_schedule], lr: flag.lr, idxd: [0],
                                    rnn_model._windInput_prop: input_prop,
                                    rnn_model._windTarget_prop: target_prop,
                                    rnn_model._windInit_prop: wInit_prop,
                                    rnn_model._windInput_vision: input_vision,
                                    rnn_model._windTarget_vision: target_vision,
                                    rnn_model._windInit_vision: wInit_vision})

                interval = time.time() - tset

                loss_step = 0.0
                loss_step_v = 0.0
                loss_step_m = 0.0
                tmp_loss, tmp_loss_v, tmp_loss_m = sess.run(rnn_model.cost_testing,
                                    feed_dict={clratio: [cl_schedule], lr: flag.lr, idxd: [0],
                                               rnn_model._windInput_prop: input_prop,
                                               rnn_model._windTarget_prop: target_prop,
                                               rnn_model._windInit_prop: wInit_prop,
                                               rnn_model._windInput_vision: input_vision,
                                               rnn_model._windTarget_vision: target_vision,
                                               rnn_model._windInit_vision: wInit_vision})
                loss_step += tmp_loss
                loss_step_v += tmp_loss_v
                loss_step_m += tmp_loss_m
                loss_step /= max_iter
                loss_step_v /= max_iter
                loss_step_m /= max_iter
                print('\rStep: %d Iter: %03d Loss: %.6f. LOSS_V: %.6f  LOSS_M: %.6f  Time/iteration: %.4f, CL: %.2f LR: %s' % (
                    t, i + 1, loss_step, loss_step_v, loss_step_m, interval, cl_schedule, flag.lr))

                tset = time.time()
                f.write("%d\t%d\t%.9f\t%.9f\t%.9f\n" % (t, i+1, loss_step, loss_step_v, loss_step_m))



                if flag.ers_adaptiveStopping:
                    if (loss_step < flag.erTH):
                        log_iteration[t] = i+1
                        break


            ## End of one epoch. Obtain the context for the next time step
            pred_prop, c_p1, c_p2, c_p3, pred_vision, c_v1, c_v2, c_v3, input_t, init_state, iidx = sess.run(rnn_model.get_pred_m,
                                               feed_dict={clratio: [cl_schedule], lr: flag.lr, idxd: [0],
                                                          rnn_model._windInput_prop: input_prop,
                                                          rnn_model._windTarget_prop: target_prop,
                                                          rnn_model._windInit_prop: wInit_prop,
                                                          rnn_model._windInput_vision: input_vision,
                                                          rnn_model._windTarget_vision: target_vision,
                                                          rnn_model._windInit_vision: wInit_vision})


            log_target_prop[0, t, :, :] = target_prop[0, -1, :, :]
            log_output_prop[0, t, :, :] = pred_prop[0, -1, :, :]

            log_target_vision[0, t, :, :] = target_vision[0, -1, :, :]
            log_output_vision[0, t, :, :] = pred_vision[0, -1, :, :]

            if (t < flag.windowLength):
                #wInit_prop[0, :, :] = pred_prop[0, 0, :, :]
                #wInit_p1[0, :] = c_p1[0, 0, 0, :]
                #wInit_p2[0, :] = c_p2[0, 0, 0, :]
                #wInit_vision[0, :, :] = pred_vision[0, 0, :, :]
                #wInit_v1[0, :, :, :] = c_v1[0, 0, 0, :, :, :]
                #wInit_v2[0, :, :, :] = c_v2[0, 0, 0, :, :, :]
                #wInit_v3[0, :, :, :] = c_v3[0, 0, 0, :, :, :]
                #wInit_p3[0, :, :, :] = c_p3[0, 0, 0, :, :, :]
                print t
            else:
                wInit_prop[0, :, :] = pred_prop[0, 0, :, :]
                wInit_p1[0, :] = c_p1[0, 1, 0, :]
                wInit_p2[0, :] = c_p2[0, 1, 0, :]
                wInit_p3[0, :, :, :] = c_p3[0, 1, 0, :, :, :]
                wInit_vision[0, :, :] = pred_vision[0, 0, :, :]
                wInit_v1[0, :, :, :] = c_v1[0, 1, 0, :, :, :]
                wInit_v2[0, :, :, :] = c_v2[0, 1, 0, :, :, :]
                wInit_v3[0, :, :, :] = c_v3[0, 1, 0, :, :, :]


            sess.run(rnn_model.set_wInitState,
                     feed_dict={rnn_model._windInit_p1_h: wInit_p1,
                                rnn_model._windInit_p2_h: wInit_p2,
                                rnn_model._windInit_p3_h: wInit_p3,
                                rnn_model._windInit_v1_h: wInit_v1,
                                rnn_model._windInit_v2_h: wInit_v2,
                                rnn_model._windInit_v3_h: wInit_v3})
            tset = time.time()




        # --------------------------------------------------------- #

        utils.save_er_logs(log_iteration, flag)
        utils.save_er_output_prop(log_output_prop, log_target_prop, flag)
        utils.save_er_output_vision(log_output_vision, log_target_vision, flag)
        utils.display_er_output_prop(log_output_prop, log_target_prop, flag)
        utils.display_er_output_vision(log_output_vision, log_target_vision, flag)


        coord.request_stop()
        coord.join(threads)

        f.close()
        sess.close()

    os.rename(flag.log_dir + "errorRegression",
              flag.log_dir + "errorRegression_type_" + str(flag.lossType) + "_wind_" + str(flag.windowLength) + "_iter_" + str(flag.numIterations) + "_" + flag.optimizer + "_lr_"+str(flag.lr)+"_id_"+str(np.random.random_sample()))

    print('Error Regression Completed.')

