import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import array_ops
import matplotlib.animation as animation

import os

import ConfigParser
Config = ConfigParser.ConfigParser()
Config.read("./settings.ini")

# To plot the proprioceptive prediction
# num = 10 indicates that 10 softmax dimensions are used to represent 1 analog value (joint angle)
# Which is equivalent to opt.out_size_smdim
softmax_array = np.linspace(-2.0, 2.0, num=10)

# For parsing the config file (settings.ini)
def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

# To reading the configuration from settings.ini
def readSettings():
    # General Settings
    tf.flags.DEFINE_string("data_dir", ConfigSectionMap("generalSettings")['data_dir'], "")
    tf.flags.DEFINE_string("data_fn", ConfigSectionMap("generalSettings")['data_fn'], "")
    tf.flags.DEFINE_string("log_dir", ConfigSectionMap("generalSettings")['log_dir'], "")
    tf.flags.DEFINE_string("log_fn", ConfigSectionMap("generalSettings")['log_fn'], "")
    tf.flags.DEFINE_string("device", ConfigSectionMap("generalSettings")['device'], "")
    tf.flags.DEFINE_bool("incremetalLearning", Config.getboolean("generalSettings", "incremetalLearning"), "")
    tf.flags.DEFINE_integer("print_epoch", Config.getint("generalSettings", "print_epoch"), "")
    tf.flags.DEFINE_integer("saveNetwork_epoch", Config.getint("generalSettings", "saveNetwork_epoch"), "")
    tf.flags.DEFINE_integer("restoreNetworkEpoch", Config.getint("generalSettings", "restoreNetworkEpoch"), "")
    tf.flags.DEFINE_float("loss_minTH", Config.getfloat("generalSettings", "loss_minTH"), "")

    # Settings for Testing (Sensory entrainment and error regression)
    tf.flags.DEFINE_integer("originalNumberOfPrimitives", Config.getint("forTesting","originalNumberOfPrimitives"), "")
    tf.flags.DEFINE_string("data_fn_testing", ConfigSectionMap("forTesting")['data_fn_testing'], "")
    ## for error regression
    tf.flags.DEFINE_integer("lossType", Config.getint("forTesting", "lossType"), "")
    tf.flags.DEFINE_integer("numIterations", Config.getint("forTesting","numIterations"), "")
    tf.flags.DEFINE_integer("windowLength", Config.getint("forTesting", "windowLength"), "")
    tf.flags.DEFINE_bool("ers_adaptiveStopping", Config.getboolean("forTesting", "ers_adaptiveStopping"), "")
    tf.flags.DEFINE_float("erTH", Config.getfloat("forTesting", "erTH"), "")

    # Learning Parameters
    tf.flags.DEFINE_integer("batch_size", Config.getint("learningParameters","batch_size"), "")
    tf.flags.DEFINE_integer("max_epoch", Config.getint("learningParameters","max_epoch"), "")
    tf.flags.DEFINE_float("lr", Config.getfloat("learningParameters","lrate"), "")
    tf.flags.DEFINE_float("cl_ratio", Config.getfloat("learningParameters", "cl_ratio"), "")
    tf.flags.DEFINE_string("optimizer", ConfigSectionMap("learningParameters")['optimizer'], "")


    # Network Settings

    # Lateral Connections
    tf.flags.DEFINE_bool("enableLateral_high_propToVision", Config.getboolean("networkSettings", "enableLateral_high_propToVision"), "")
    tf.flags.DEFINE_bool("enableLateral_high_visionToProp", Config.getboolean("networkSettings", "enableLateral_high_visionToProp"), "")
    tf.flags.DEFINE_bool("enableLateral_mid_propToVision", Config.getboolean("networkSettings", "enableLateral_mid_propToVision"), "")
    tf.flags.DEFINE_bool("enableLateral_mid_visionToProp", Config.getboolean("networkSettings", "enableLateral_mid_visionToProp"), "")
    tf.flags.DEFINE_bool("enableLateral_low_propToVision", Config.getboolean("networkSettings", "enableLateral_low_propToVision"), "")
    tf.flags.DEFINE_bool("enableLateral_low_visionToProp", Config.getboolean("networkSettings", "enableLateral_low_visionToProp"), "")

    # Network Settings for Prop. Pathway
    tf.flags.DEFINE_integer("p1_unit", Config.getint("networkSettings", "p1_unit"), "")
    tf.flags.DEFINE_integer("p2_unit", Config.getint("networkSettings", "p2_unit"), "")
    tf.flags.DEFINE_integer("p3_unit", Config.getint("networkSettings", "p3_unit"), "")
    tf.flags.DEFINE_float("p1_tau", Config.getfloat("networkSettings", "p1_tau"), "")
    tf.flags.DEFINE_float("p2_tau", Config.getfloat("networkSettings", "p2_tau"), "")
    tf.flags.DEFINE_float("p3_tau", Config.getfloat("networkSettings", "p3_tau"), "")

    # Network Settings for Visual Pathway
    tf.flags.DEFINE_integer("v1_unit", Config.getint("networkSettings", "v1_unit"), "")
    tf.flags.DEFINE_integer("v2_unit", Config.getint("networkSettings", "v2_unit"), "")
    tf.flags.DEFINE_integer("v3_unit", Config.getint("networkSettings", "v3_unit"), "")
    tf.flags.DEFINE_float("v1_tau", Config.getfloat("networkSettings", "v1_tau"), "")
    tf.flags.DEFINE_float("v2_tau", Config.getfloat("networkSettings", "v2_tau"), "")
    tf.flags.DEFINE_float("v3_tau", Config.getfloat("networkSettings", "v3_tau"), "")

    tf.flags.DEFINE_integer("v1_convFilter_h", Config.getint("networkSettings", "v1_convFilter_h"), "")
    tf.flags.DEFINE_integer("v1_convFilter_w", Config.getint("networkSettings", "v1_convFilter_w"), "")
    tf.flags.DEFINE_integer("v1_convStride_h", Config.getint("networkSettings", "v1_convStride_h"), "")
    tf.flags.DEFINE_integer("v1_convStride_w", Config.getint("networkSettings", "v1_convStride_w"), "")

    tf.flags.DEFINE_integer("v2_convFilter_h", Config.getint("networkSettings", "v2_convFilter_h"), "")
    tf.flags.DEFINE_integer("v2_convFilter_w", Config.getint("networkSettings", "v2_convFilter_w"), "")
    tf.flags.DEFINE_integer("v2_convStride_h", Config.getint("networkSettings", "v2_convStride_h"), "")
    tf.flags.DEFINE_integer("v2_convStride_w", Config.getint("networkSettings", "v2_convStride_w"), "")

    tf.flags.DEFINE_integer("v3_convFilter_h", Config.getint("networkSettings", "v3_convFilter_h"), "")
    tf.flags.DEFINE_integer("v3_convFilter_w", Config.getint("networkSettings", "v3_convFilter_w"), "")
    tf.flags.DEFINE_integer("v3_convStride_h", Config.getint("networkSettings", "v3_convStride_h"), "")
    tf.flags.DEFINE_integer("v3_convStride_w", Config.getint("networkSettings", "v3_convStride_w"), "")

    tf.flags.DEFINE_integer("p3_convFilter_h", Config.getint("networkSettings", "p3_convFilter_h"), "")
    tf.flags.DEFINE_integer("p3_convFilter_w", Config.getint("networkSettings", "p3_convFilter_w"), "")
    tf.flags.DEFINE_integer("p3_convStride_h", Config.getint("networkSettings", "p3_convStride_h"), "")
    tf.flags.DEFINE_integer("p3_convStride_w", Config.getint("networkSettings", "p3_convStride_w"), "")

    tf.flags.DEFINE_integer("v1_msize_h", Config.getint("networkSettings","v1_msize_h"), "")
    tf.flags.DEFINE_integer("v1_msize_w", Config.getint("networkSettings","v1_msize_w"), "")
    tf.flags.DEFINE_integer("v2_msize_h", Config.getint("networkSettings","v2_msize_h"), "")
    tf.flags.DEFINE_integer("v2_msize_w", Config.getint("networkSettings","v2_msize_w"), "")
    tf.flags.DEFINE_integer("v3_msize_h", Config.getint("networkSettings","v3_msize_h"), "")
    tf.flags.DEFINE_integer("v3_msize_w", Config.getint("networkSettings","v3_msize_w"), "")
    tf.flags.DEFINE_integer("p3_msize_h", Config.getint("networkSettings","p3_msize_h"), "")
    tf.flags.DEFINE_integer("p3_msize_w", Config.getint("networkSettings","p3_msize_w"), "")

    flag = tf.flags.FLAGS
    flag.v1_msize = (Config.getint("networkSettings","v1_msize_h"), Config.getint("networkSettings","v1_msize_w"))
    flag.v2_msize = (Config.getint("networkSettings","v2_msize_h"), Config.getint("networkSettings","v2_msize_w"))
    flag.v3_msize = (Config.getint("networkSettings","v3_msize_h"), Config.getint("networkSettings","v3_msize_w"))
    flag.p3_msize = (Config.getint("networkSettings","p3_msize_h"), Config.getint("networkSettings","p3_msize_w"))

    flag.v1_size = flag.v1_msize[0] * flag.v1_msize[1]
    flag.v2_size = flag.v2_msize[0] * flag.v2_msize[1]
    flag.v3_size = flag.v3_msize[0] * flag.v3_msize[1]
    flag.p3_size = flag.p3_msize[0] * flag.p3_msize[1]


    return flag


def readDataset(type):
    #Type 0: Training
    #Type 1: Testing (Error Regression & Sensory Entrainment)

    # Read the model settings (options)
    flag = readSettings()

    print('=' * 100)
    print "Loading the dataset..."

    class DataSets(object):
        pass

    dbs = DataSets()
    if type == 0:
        data_raw = np.load(flag.data_dir + flag.data_fn)
    else:
        data_raw = np.load(flag.data_dir + flag.data_fn_testing)


    print('-' * 100)
    dbs.prop = data_raw['motor']
    print "Prop. shape:", dbs.prop.shape
    print"Number of data : ", dbs.prop.shape[0]
    flag.length = dbs.prop.shape[1]
    print "Length :", flag.length
    flag.out_size_mdim = dbs.prop.shape[-2]
    print"Analog dimension : ", flag.out_size_mdim
    flag.out_size_smdim = dbs.prop.shape[-1]
    print"Softmax dimension : ", flag.out_size_smdim
    print('-' * 100)
    dbs.vision = data_raw['vision']
    flag.out_size_vrow = dbs.vision.shape[-2]
    flag.out_size_vcol = dbs.vision.shape[-1]
    print "Vision shape: ", dbs.vision.shape
    dbs.idxd = data_raw['idxd']
    print"Number of data : ", dbs.vision.shape[0]
    flag.length = dbs.vision.shape[1] - 1  # due to 1-step prediction
    print "Length :", flag.length
    print "image height :", dbs.vision.shape[2]
    print "image width   :", dbs.vision.shape[3]
    print('-' * 100)
    print "idxd shape: ", dbs.idxd.shape
    print "idxd: ", dbs.idxd

    if type == 0:
        flag.isThisTrain = True  # Set this true in case of training & checkTrainingResult
        flag.num_data = dbs.idxd.shape[0]
    else:
        flag.isThisTrain = False  # Set this true in case of training & checkTrainingResult
        flag.num_data = flag.originalNumberOfPrimitives
        flag.batch_size = 1


    print('=' * 100)
    try:
        input("Press enter to continue")
    except SyntaxError:
        pass

    # Check the batch size
    if (flag.batch_size > flag.num_data):
        print "Please check the batch size."
        assert flag.num_data >= flag.batch_size

    return flag, dbs


# To save the training results (input/output/neuron activation)
def save_results(sess, model, opt):
    # ========================================================================
    # Saving the initial states obtained from training.
    # The initial states of each layer for each training pattern will be saved in ./log_dir/initStates/
    # ========================================================================
    print "=" * 100
    print "Saving the initial states."
    isdir = os.path.exists(opt.log_dir+"initStates")
    if not isdir:
        os.makedirs(opt.log_dir+"initStates")

    p1_initStates, p2_initStates, p3_initStates, v1_initStates, v2_initStates, v3_initStates  = sess.run(model.get_initStates)

    for idxSeq in xrange(opt.num_data):
        print "Saving the initial states of the training data: ", idxSeq

        fid_o_p1 = open(opt.log_dir + "initStates" + "/init_p1_%d.txt" % idxSeq, 'w')
        fid_o_p2 = open(opt.log_dir + "initStates" + "/init_p2_%d.txt" % idxSeq, 'w')
        fid_o_p3 = open(opt.log_dir + "initStates" + "/init_p3_%d.txt" % idxSeq, 'w')

        fid_o_v1 = open(opt.log_dir+"initStates"+"/init_v1_%d.txt" % idxSeq, 'w')
        fid_o_v2 = open(opt.log_dir+"initStates"+"/init_v2_%d.txt" % idxSeq, 'w')
        fid_o_v3 = open(opt.log_dir+"initStates"+"/init_v3_%d.txt" % idxSeq, 'w')


        for fm in xrange(opt.p1_unit):
            data = "%f\t" % p1_initStates[idxSeq, fm]
            fid_o_p1.write(data)
        fid_o_p1.write("\n")

        for fm in xrange(opt.p2_unit):
            data = "%f\t" % p2_initStates[idxSeq, fm]
            fid_o_p2.write(data)
        fid_o_p2.write("\n")

        for fm in xrange(opt.p3_unit):
            for height in xrange(opt.p3_msize[0]):
                for width in xrange(opt.p3_msize[1]):
                    data = "%f\t" % p3_initStates[idxSeq, height, width, fm]
                    fid_o_p3.write(data)
        fid_o_p3.write("\n")

        for fm in xrange(opt.v1_unit):
            for height in xrange(opt.v1_msize[0]):
                for width in xrange(opt.v1_msize[1]):
                    data = "%f\t" % v1_initStates[idxSeq, height, width, fm]
                    fid_o_v1.write(data)
        fid_o_v1.write("\n")

        for fm in xrange(opt.v2_unit):
            for height in xrange(opt.v2_msize[0]):
                for width in xrange(opt.v2_msize[1]):
                    data = "%f\t" % v2_initStates[idxSeq, height, width, fm]
                    fid_o_v2.write(data)
        fid_o_v2.write("\n")

        for fm in xrange(opt.v3_unit):
            for height in xrange(opt.v3_msize[0]):
                for width in xrange(opt.v3_msize[1]):
                    data = "%f\t" % v3_initStates[idxSeq, height, width, fm]
                    fid_o_v3.write(data)
        fid_o_v3.write("\n")


        fid_o_p1.close()
        fid_o_p2.close()
        fid_o_v1.close()
        fid_o_v2.close()
        fid_o_v3.close()
        fid_o_p3.close()

    #========================================================================
    # Saving the input / output / neuron activation
    # ========================================================================
    print "=" * 100
    print "Saving the input, output, neuron activation.."

    isdir = os.path.exists(opt.log_dir + "outputs")
    if not isdir:
        os.makedirs(opt.log_dir + "outputs")

    out_size_vrow = opt.out_size_vrow
    out_size_vcol = opt.out_size_vcol
    numData = opt.num_data
    length = opt.length

    image_input_openLoop = np.random.rand(opt.num_data, length, out_size_vrow, out_size_vcol)
    image_input_closedLoop = np.random.rand(opt.num_data, length, out_size_vrow, out_size_vcol)
    image_pred_openLoop = np.random.rand(opt.num_data, length, out_size_vrow, out_size_vcol)
    image_pred_closedLoop = np.random.rand(opt.num_data, length, out_size_vrow, out_size_vcol)

    prop_input_openLoop = np.random.rand(numData, length, opt.out_size_mdim, opt.out_size_smdim)
    prop_input_closedLoop = np.random.rand(numData, length, opt.out_size_mdim, opt.out_size_smdim)
    prop_pred_openLoop = np.random.rand(numData, length, opt.out_size_mdim, opt.out_size_smdim)
    prop_pred_closedLoop = np.random.rand(numData, length, opt.out_size_mdim, opt.out_size_smdim)

    layer_p1_closed = np.random.rand(opt.num_data, length, 2, opt.p1_unit)
    layer_p2_closed = np.random.rand(opt.num_data, length, 2, opt.p2_unit)
    layer_p3_closed = np.random.rand(opt.num_data, length, 2, opt.p3_msize[0], opt.p3_msize[1], opt.p3_unit)

    layer_p1_open = np.random.rand(opt.num_data, length, 2, opt.p1_unit)
    layer_p2_open = np.random.rand(opt.num_data, length, 2, opt.p2_unit)
    layer_p3_open = np.random.rand(opt.num_data, length, 2, opt.p3_msize[0], opt.p3_msize[1], opt.p3_unit)

    layer_v1_closed = np.random.rand(opt.num_data, length, 2, opt.v1_msize[0], opt.v1_msize[1], opt.v1_unit)
    layer_v2_closed = np.random.rand(opt.num_data, length, 2, opt.v2_msize[0], opt.v2_msize[1], opt.v2_unit)
    layer_v3_closed = np.random.rand(opt.num_data, length, 2, opt.v3_msize[0], opt.v3_msize[1], opt.v3_unit)

    layer_v1_open = np.random.rand(opt.num_data, length, 2, opt.v1_msize[0], opt.v1_msize[1], opt.v1_unit)
    layer_v2_open = np.random.rand(opt.num_data, length, 2, opt.v2_msize[0], opt.v2_msize[1], opt.v2_unit)
    layer_v3_open = np.random.rand(opt.num_data, length, 2, opt.v3_msize[0], opt.v3_msize[1], opt.v3_unit)


    numTrial = 100
    check_oIdx = np.ones(opt.num_data)*9999
    check_cIdx = np.ones(opt.num_data) * 9999
    openCompleted = 0
    closedCompleted = 0

    for iterations in xrange(numTrial):
        print ('Checking the index.. trial: %s' % (iterations+1))
        pred_m_open, open_p1, open_p2, open_p3, pred_v_open, open_v1, open_v2, open_v3, input_t_open, init_state, idxd_open = sess.run(
            model.prediction_pmstrnn, feed_dict={model._cl: [0.0], model._lr: 0.01})
        pred_m_closed, closed_p1, closed_p2, closed_p3, pred_v_closed, closed_v1, closed_v2, closed_v3, input_t_closed, init_state, idxd_closed = sess.run(
            model.prediction_pmstrnn, feed_dict={model._cl: [1.0], model._lr: 0.01})
        
        prop_t_open, vision_t_open = input_t_open
        prop_t_closed, vision_t_closed = input_t_closed

        for iii in xrange(opt.num_data):
            o_idx = idxd_open[iii]
            c_idx = idxd_closed[iii]

            prop_input_openLoop[o_idx, :, :, :] = prop_t_open[:, iii, :, :]
            prop_pred_openLoop[o_idx, :, :, :] = pred_m_open[iii, :, :, :]

            prop_input_closedLoop[c_idx, :, :, :] = prop_t_closed[:, iii, :, :]
            prop_pred_closedLoop[c_idx, :, :, :] = pred_m_closed[iii, :, :, :]

            image_input_openLoop[o_idx, :, :, :] = vision_t_open[:, iii, :, :]
            image_pred_openLoop[o_idx, :, :, :] = pred_v_open[iii, :, :, :]

            image_input_closedLoop[c_idx, :, :, :] = vision_t_closed[:, iii, :, :]
            image_pred_closedLoop[c_idx, :, :, :] = pred_v_closed[iii, :, :, :]

            layer_p1_closed[c_idx, :, :, :] = closed_p1[iii, :, :, :]
            layer_p2_closed[c_idx, :, :, :] = closed_p2[iii, :, :, :]
            layer_p3_closed[c_idx, :, :, :, :, :] = closed_p3[iii, :, :, :, :, :]
            layer_v1_closed[c_idx, :, :, :, :, :] = closed_v1[iii, :, :, :, :, :]
            layer_v2_closed[c_idx, :, :, :, :, :] = closed_v2[iii, :, :, :, :, :]
            layer_v3_closed[c_idx, :, :, :, :, :] = closed_v3[iii, :, :, :, :, :]

            layer_p1_open[o_idx, :, :, :] = open_p1[iii, :, :, :]
            layer_p2_open[o_idx, :, :, :] = open_p2[iii, :, :, :]
            layer_p3_open[o_idx, :, :, :, :, :] = open_p3[iii, :, :, :, :, :]
            layer_v1_open[o_idx, :, :, :, :, :] = open_v1[iii, :, :, :, :, :]
            layer_v2_open[o_idx, :, :, :, :, :] = open_v2[iii, :, :, :, :, :]
            layer_v3_open[o_idx, :, :, :, :, :] = open_v3[iii, :, :, :, :, :]

            check_oIdx[o_idx] = idxd_open[iii]
            check_cIdx[c_idx] = idxd_closed[iii]

        if np.amax(check_oIdx) != 9999:
            openCompleted = 1
        if np.amax(check_cIdx) != 9999:
            closedCompleted = 1

        print('\rOpen-loop Index: %s' % (check_oIdx))
        print('\rClosed-loop Index: %s' % (check_cIdx))
        if (openCompleted and closedCompleted):
            break



    for idxSeq in xrange(opt.num_data):
        print "Saving the results for the training data: ", idxSeq

        f = open(opt.log_dir + "outputs" + "/input_prop_%d.txt" % idxSeq, 'w')
        f2 = open(opt.log_dir + "outputs" + "/open_prop_%d.txt" % idxSeq, 'w')
        f3 = open(opt.log_dir + "outputs" + "/closed_prop_%d.txt" % idxSeq, 'w')
        for step in xrange(length):
            for analogDim in xrange(opt.out_size_mdim):
                for softmaxDim in xrange(opt.out_size_smdim):
                    data = "%f\t" % prop_input_openLoop[idxSeq, step, analogDim, softmaxDim]
                    f.write(data)
                    data2 = "%f\t" % prop_pred_openLoop[idxSeq, step, analogDim, softmaxDim]
                    f2.write(data2)
                    data3 = "%f\t" % prop_pred_closedLoop[idxSeq, step, analogDim, softmaxDim]
                    f3.write(data3)
            f.write("\n")
            f2.write("\n")
            f3.write("\n")
        f.close()
        f2.close()
        f3.close()

        f = open(opt.log_dir + "outputs" + "/input_vision_%d.txt" % idxSeq, 'w')
        f2 = open(opt.log_dir + "outputs" + "/open_vision_%d.txt" % idxSeq, 'w')
        f3 = open(opt.log_dir + "outputs" + "/closed_vision_%d.txt" % idxSeq, 'w')
        for step in xrange(length):
            for height in xrange(out_size_vrow):
                for width in xrange(out_size_vcol):
                    data = "%f\t" % image_input_openLoop[idxSeq, step, height, width]
                    f.write(data)
                    data2 = "%f\t" % image_pred_openLoop[idxSeq, step, height, width]
                    f2.write(data2)
                    data3 = "%f\t" % image_pred_closedLoop[idxSeq, step, height, width]
                    f3.write(data3)
            f.write("\n")
            f2.write("\n")
            f3.write("\n")
        f.close()
        f2.close()
        f3.close()

        fid_closed_p1 = open(opt.log_dir + "outputs" + "/closed_p1_%d.txt" % idxSeq, 'w')
        fid_closed_p2 = open(opt.log_dir + "outputs" + "/closed_p2_%d.txt" % idxSeq, 'w')
        fid_closed_p3 = open(opt.log_dir + "outputs" + "/closed_p3_%d.txt" % idxSeq, 'w')
        fid_closed_v1 = open(opt.log_dir + "outputs" + "/closed_v1_%d.txt" % idxSeq, 'w')
        fid_closed_v2 = open(opt.log_dir + "outputs" + "/closed_v2_%d.txt" % idxSeq, 'w')
        fid_closed_v3 = open(opt.log_dir + "outputs" + "/closed_v3_%d.txt" % idxSeq, 'w')

        fid_o_p1 = open(opt.log_dir + "outputs" + "/open_p1_%d.txt" % idxSeq, 'w')
        fid_o_p2 = open(opt.log_dir + "outputs" + "/open_p2_%d.txt" % idxSeq, 'w')
        fid_o_p3 = open(opt.log_dir + "outputs" + "/open_p3_%d.txt" % idxSeq, 'w')
        fid_o_v1 = open(opt.log_dir + "outputs" + "/open_v1_%d.txt" % idxSeq, 'w')
        fid_o_v2 = open(opt.log_dir + "outputs" + "/open_v2_%d.txt" % idxSeq, 'w')
        fid_o_v3 = open(opt.log_dir + "outputs" + "/open_v3_%d.txt" % idxSeq, 'w')

        for step in xrange(length):

            # Closed loop result
            for fm in xrange(opt.p1_unit):
                data = "%f\t" % layer_p1_closed[idxSeq, step, 1, fm]
                fid_closed_p1.write(data)
            fid_closed_p1.write("\n")

            for fm in xrange(opt.p2_unit):
                data = "%f\t" % layer_p2_closed[idxSeq, step, 1, fm]
                fid_closed_p2.write(data)
            fid_closed_p2.write("\n")

            for fm in xrange(opt.v1_unit):
                for height in xrange(opt.v1_msize[0]):
                    for width in xrange(opt.v1_msize[1]):
                        data = "%f\t" % layer_v1_closed[idxSeq, step, 1, height, width, fm]
                        fid_closed_v1.write(data)
            fid_closed_v1.write("\n")

            for fm in xrange(opt.v2_unit):
                for height in xrange(opt.v2_msize[0]):
                    for width in xrange(opt.v2_msize[1]):
                        data = "%f\t" % layer_v2_closed[idxSeq, step, 1, height, width, fm]
                        fid_closed_v2.write(data)
            fid_closed_v2.write("\n")

            for fm in xrange(opt.v3_unit):
                for height in xrange(opt.v3_msize[0]):
                    for width in xrange(opt.v3_msize[1]):
                        data = "%f\t" % layer_v3_closed[idxSeq, step, 1, height, width, fm]
                        fid_closed_v3.write(data)
            fid_closed_v3.write("\n")

            for fm in xrange(opt.p3_unit):
                for height in xrange(opt.p3_msize[0]):
                    for width in xrange(opt.p3_msize[1]):
                        data = "%f\t" % layer_p3_closed[idxSeq, step, 1, height, width, fm]
                        fid_closed_p3.write(data)
            fid_closed_p3.write("\n")


            # Open loop result
            for fm in xrange(opt.p1_unit):
                data = "%f\t" % layer_p1_open[idxSeq, step, 1, fm]
                fid_o_p1.write(data)
            fid_o_p1.write("\n")

            for fm in xrange(opt.p2_unit):
                data = "%f\t" % layer_p2_open[idxSeq, step, 1, fm]
                fid_o_p2.write(data)
            fid_o_p2.write("\n")

            for fm in xrange(opt.p3_unit):
                for height in xrange(opt.p3_msize[0]):
                    for width in xrange(opt.p3_msize[1]):
                        data = "%f\t" % layer_p3_open[idxSeq, step, 1, height, width, fm]
                        fid_o_p3.write(data)
            fid_o_p3.write("\n")

            for fm in xrange(opt.v1_unit):
                for height in xrange(opt.v1_msize[0]):
                    for width in xrange(opt.v1_msize[1]):
                        data = "%f\t" % layer_v1_open[idxSeq, step, 1, height, width, fm]
                        fid_o_v1.write(data)
            fid_o_v1.write("\n")

            for fm in xrange(opt.v2_unit):
                for height in xrange(opt.v2_msize[0]):
                    for width in xrange(opt.v2_msize[1]):
                        data = "%f\t" % layer_v2_open[idxSeq, step, 1, height, width, fm]
                        fid_o_v2.write(data)
            fid_o_v2.write("\n")

            for fm in xrange(opt.v3_unit):
                for height in xrange(opt.v3_msize[0]):
                    for width in xrange(opt.v3_msize[1]):
                        data = "%f\t" % layer_v3_open[idxSeq, step, 1, height, width, fm]
                        fid_o_v3.write(data)
            fid_o_v3.write("\n")


        fid_closed_p1.close()
        fid_closed_p2.close()
        fid_closed_p3.close()
        fid_closed_v1.close()
        fid_closed_v2.close()
        fid_closed_v3.close()

        fid_o_p1.close()
        fid_o_p2.close()
        fid_o_p3.close()
        fid_o_v1.close()
        fid_o_v2.close()
        fid_o_v3.close()


# To plot proprioceptive & visual target/open-loop output/closed-loop output after training
def plot_outputs(sess, model, opt):
    print "=" * 100
    print "Drawing the inputs & outputs.."
    out_size_vrow = opt.out_size_vrow
    out_size_vcol = opt.out_size_vcol
    numData = opt.num_data
    length = opt.length

    image_input_openLoop = np.random.rand(numData, length, out_size_vrow, out_size_vcol)
    image_input_closedLoop = np.random.rand(numData, length, out_size_vrow, out_size_vcol)
    image_pred_openLoop = np.random.rand(numData, length, out_size_vrow, out_size_vcol)
    image_pred_closedLoop = np.random.rand(numData, length, out_size_vrow, out_size_vcol)

    prop_pred_openLoop = np.random.rand(numData, length, opt.out_size_mdim, opt.out_size_smdim)
    prop_pred_closedLoop = np.random.rand(numData, length, opt.out_size_mdim, opt.out_size_smdim)
    prop_target = np.random.rand(length, numData, opt.out_size_mdim, opt.out_size_smdim)

    numTrial = 100
    check_oIdx = np.ones(opt.num_data)*9999
    check_cIdx = np.ones(opt.num_data) * 9999
    openCompleted = 0
    closedCompleted = 0

    for iterations in xrange(numTrial):
        print ('Checking the index.. trial: %s' % (iterations + 1))
        pred_m_open, open_p1, open_p2, open_p3, pred_v_open, open_v1, open_v2, open_v3, input_t_open, init_state, idxd_open = sess.run(
            model.prediction_pmstrnn, feed_dict={model._cl: [0.0], model._lr: 0.01})
        input_m_open, input_v_open = input_t_open

        pred_m_closed, closed_p1, closed_p2, closed_p3, pred_v_closed, closed_v1, closed_v2, closed_v3, input_t_closed, init_state, idxd_closed = sess.run(
            model.prediction_pmstrnn, feed_dict={model._cl: [1.0], model._lr: 0.01})
        input_m_closed, input_v_closed = input_t_closed

        for iii in xrange(numData):
            o_idx = idxd_open[iii]
            c_idx = idxd_closed[iii]

            image_input_openLoop[o_idx, :, :, :] = input_v_open[:, iii, :, :]
            image_pred_openLoop[o_idx, :, :, :] = pred_v_open[iii, :, :, :]
            prop_pred_openLoop[o_idx, :, :, :] = pred_m_open[iii, :, :, :]

            image_input_closedLoop[c_idx, :, :, :] = input_v_closed[:, iii, :, :]
            image_pred_closedLoop[c_idx, :, :, :] = pred_v_closed[iii, :, :, :]
            prop_pred_closedLoop[c_idx, :, :, :] = pred_m_closed[iii, :, :, :]

            prop_target[: , c_idx, :, :] = input_m_closed[:, iii, :, :]

            check_oIdx[o_idx] = idxd_open[iii]
            check_cIdx[c_idx] = idxd_closed[iii]

        if np.amax(check_oIdx) != 9999:
            openCompleted = 1
        if np.amax(check_cIdx) != 9999:
            closedCompleted = 1

        print('\rOpen-loop Index: %s' % (check_oIdx))
        print('\rClosed-loop Index: %s' % (check_cIdx))
        if (openCompleted and closedCompleted):
            break


    analog_dim = prop_target.shape[2]
    softmax_dim = opt.out_size_smdim
    for idxImg in xrange(numData):
        # For Proprioceptive Output
        analog_pred_open = 0
        analog_pred_closed = 0
        analog_target = 0
        for a in range(softmax_dim):
            analog_pred_open = analog_pred_open + prop_pred_openLoop[idxImg, :, :, a] * softmax_array[a]
            analog_pred_closed = analog_pred_closed + prop_pred_closedLoop[idxImg, :, :, a] * softmax_array[a]
            analog_target = analog_target + prop_target[: , idxImg, :, a] * softmax_array[a]

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # Shape
        # pred_m_open:  (numData, Length, analogDim, softmaxDim)
        # e.g.) pred_m_open:  (2, 140, 2, 10)
        # input_m_open:  (length, numData, analogDim, softmaxDim)
        # e.g.) input_m_open:  (140, 2, 2, 10)
        # analog_pred_open: (numData, Length, analogDim)
        # e.g.) analog_pred_open: (2, 140, 2)
        # analog_target: (length, numData, analogdim)
        # e.g.) analog_target: (140, 2, 2)

        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, analog_dim)]

        for j in range(analog_dim):
            #ax1.plot(analog_pred_open[0:-2, j], 'b-', label='predictions')
            #label = '$y = {i}x + {i}$'.format(i=i)
            ax1.plot(analog_pred_open[0:-2, j], color=colors[j], label = 'prediction')
            ax1.plot(analog_target[1:-1, j], color=colors[j], linestyle = '--', label='target')
            ax1.set_xlabel('open loop')

            ax2.plot(analog_pred_closed[0:-2, j], color=colors[j], label = 'prediction')
            ax2.plot(analog_target[1:-1, j], color=colors[j], linestyle = '--', label='target')
            ax2.set_xlabel('closed loop')

        plt.show()

        # ------------------------------------------------------------------------------------------------#


        # For Vision Output
        # setup figure
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        plt.set_cmap('gray')

        ax1.set_xlabel('Target')
        ax2.set_xlabel('Open-loop')
        ax3.set_xlabel('Closed-loop')

        # set up list of images for animation
        ims = []
        for time in xrange(length):
            im = ax1.imshow(image_input_openLoop[idxImg, time, :, :])
            im2 = ax2.imshow(image_pred_openLoop[idxImg, time, :, :])
            im3 = ax3.imshow(image_pred_closedLoop[idxImg, time, :, :])
            ims.append([im, im2, im3])

        # run animation
        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat=True)
        plt.show()





# To save the result from the testing - sensory entrainment
def save_entrainment_logs(input_t, log_prop, log_p1, log_p2, log_p3, log_vision, log_v1, log_v2, log_v3, opt):
    print "=" * 100
    print "Saving the sensory entrainment results - input, output, feature maps.."
    isdir = os.path.exists(opt.log_dir + "entrainment")
    if not isdir:
        os.makedirs(opt.log_dir + "entrainment")

    length = opt.length
    m_t, v_t = input_t
    # m_t.shape: (length, numData, analogDim, softmaxDim)
    # e.g.) m_t:  (321, 1, 2, 10)
    # v_t.shape: (length, numData, height, width)
    # e.g.) v_t:  (321, 1, 48, 64)

    # pred_vision shape: (numData, length, height, width)
    # e.g.) pred_vision: (1, 321, 48, 64)
    # pred_prop: (numData, length, analogDim, softmaxDim)
    # e.g.) pred_prop: (1, 321, 2, 10)
    # c_p1 shape: (numData, length, 2(h & y), numUnit)
    # e.g.) c_p1: (1, 321, 2, 30)
    # c_v1 shape: (numData, length, 2(h & y), height, width, numUnit)
    # e.g.) c_v1: (1, 321, 2, 44, 60, 8)

    out_size_vrow = opt.out_size_vrow
    out_size_vcol = opt.out_size_vcol


    fid_prop_pred = open(opt.log_dir + "entrainment" + "/prop_output.txt", 'w')
    fid_prop_target = open(opt.log_dir + "entrainment" + "/prop_target.txt", 'w')

    fid_vision_pred = open(opt.log_dir + "entrainment" + "/vision_output.txt", 'w')
    fid_vision_target = open(opt.log_dir + "entrainment" + "/vision_target.txt", 'w')

    fid_c_p1 = open(opt.log_dir + "entrainment" + "/ctx_p1.txt", 'w')
    fid_c_p2 = open(opt.log_dir + "entrainment" + "/ctx_p2.txt", 'w')
    fid_c_p3 = open(opt.log_dir + "entrainment" + "/ctx_p3.txt", 'w')
    fid_c_v1 = open(opt.log_dir + "entrainment" + "/ctx_v1.txt", 'w')
    fid_c_v2 = open(opt.log_dir + "entrainment" + "/ctx_v2.txt", 'w')
    fid_c_v3 = open(opt.log_dir + "entrainment" + "/ctx_v3.txt", 'w')
    

    for step in xrange(length):

        for analogDim in xrange(opt.out_size_mdim):
            for softmaxDim in xrange(opt.out_size_smdim):
                data = "%f\t" % m_t[step, 0, analogDim, softmaxDim]
                fid_prop_target.write(data)
        fid_prop_target.write("\n")

        for analogDim in xrange(opt.out_size_mdim):
            for softmaxDim in xrange(opt.out_size_smdim):
                data = "%f\t" % log_prop[0, step, analogDim, softmaxDim]
                fid_prop_pred.write(data)
        fid_prop_pred.write("\n")

        for height in xrange(out_size_vrow):
            for width in xrange(out_size_vcol):
                data = "%f\t" % v_t[step, 0, height, width]
                fid_vision_target.write(data)
        fid_vision_target.write("\n")

        for height in xrange(out_size_vrow):
            for width in xrange(out_size_vcol):
                data = "%f\t" % log_vision[0, step, height, width]
                fid_vision_pred.write(data)
        fid_vision_pred.write("\n")

        for fm in xrange(opt.p1_unit):
            data = "%f\t" % log_p1[0, step, 1, fm]
            fid_c_p1.write(data)
        fid_c_p1.write("\n")

        for fm in xrange(opt.p2_unit):
            data = "%f\t" % log_p2[0, step, 1, fm]
            fid_c_p2.write(data)
        fid_c_p2.write("\n")

        for fm in xrange(opt.p3_unit):
            for height in xrange(opt.p3_msize[0]):
                for width in xrange(opt.p3_msize[1]):
                    data = "%f\t" % log_p3[0, step, 1, height, width, fm]
                    fid_c_p3.write(data)
        fid_c_p3.write("\n")

        for fm in xrange(opt.v1_unit):
            for height in xrange(opt.v1_msize[0]):
                for width in xrange(opt.v1_msize[1]):
                    data = "%f\t" % log_v1[0, step, 1, height, width, fm]
                    fid_c_v1.write(data)
        fid_c_v1.write("\n")

        for fm in xrange(opt.v2_unit):
            for height in xrange(opt.v2_msize[0]):
                for width in xrange(opt.v2_msize[1]):
                    data = "%f\t" % log_v2[0, step, 1, height, width, fm]
                    fid_c_v2.write(data)
        fid_c_v2.write("\n")

        for fm in xrange(opt.v3_unit):
            for height in xrange(opt.v3_msize[0]):
                for width in xrange(opt.v3_msize[1]):
                    data = "%f\t" % log_v3[0, step, 1, height, width, fm]
                    fid_c_v3.write(data)
        fid_c_v3.write("\n")




    fid_prop_pred.close()
    fid_vision_pred.close()
    fid_prop_target.close()
    fid_vision_target.close()
    fid_c_p1.close()
    fid_c_p2.close()
    fid_c_p3.close()
    fid_c_v1.close()
    fid_c_v2.close()
    fid_c_v3.close()

# To plot the result from the testing - sensory entrainment
def display_entrainment_logs(input_t, log_prop, log_p1, log_p2, log_p3, log_vision, log_v1, log_v2, log_v3, opt):
    print "=" * 100
    print "Displaying the sensory entrainment results - target & output.."
    isdir = os.path.exists(opt.log_dir + "entrainment")
    if not isdir:
        os.makedirs(opt.log_dir + "entrainment")

    length = opt.length
    m_t, v_t = input_t


    # For Prop Output _ Open
    analog_pred = 0
    analog_target = 0
    for a in range(opt.out_size_smdim):
        analog_target = analog_target + m_t[:, 0, :, a] * softmax_array[a]
        analog_pred = analog_pred + log_prop[0, :, :, a] * softmax_array[a]

    # fig, ax = plt.subplots(nrows=2, sharex=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    analog_dim = analog_pred.shape[1]
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, analog_dim)]
    for j in range(analog_dim):
        ax1.plot(analog_pred[0:-2, j], color=colors[j], label='prediction')
        ax1.plot(analog_target[1:-1, j], color=colors[j], linestyle='--', label='target')
        ax1.set_xlabel('Error Regression Closed-loop Prop.')

    plt.show()

    # For Vision Output
    # setup figure
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.set_cmap('gray')

    ax1.set_xlabel('Target')
    ax2.set_xlabel('Sensory Entrainment')
    # ax3.set_xlabel('Target')

    # v_t:  (length, 1, height, width)
    # pred_v: (1, length, height, width)

    # set up list of images for animation
    ims = []
    for time in xrange(length):
        im = ax1.imshow(v_t[time, 0, :, :])
        im2 = ax2.imshow(log_vision[0, time, :, :])
        # im3 = ax3.imshow(image_ic[idxImg, time, :, :])
        ims.append([im, im2])

    # run animation
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat=True)
    plt.show()




# Saving the number of iterations during the ERS
def save_er_logs(totalIter, opt):
    isdir = os.path.exists(opt.log_dir + "errorRegression")
    if not isdir:
        os.makedirs(opt.log_dir + "errorRegression")

    f = open(opt.log_dir + "errorRegression" + "/numberOfIterations.txt", 'w')
    for step in xrange(opt.length):
        data = "%d\n" % totalIter[step]
        f.write(data)
    f.close()

# To save the propOutput after the ERS
def save_er_output_prop(outputs, targets, opt):
    isdir = os.path.exists(opt.log_dir + "errorRegression")
    if not isdir:
        os.makedirs(opt.log_dir + "errorRegression")

    f = open(opt.log_dir + "errorRegression" + "/target_prop_er.txt", 'w')
    f3 = open(opt.log_dir + "errorRegression" + "/closed_errorRegression_prop.txt", 'w')
    for step in xrange(opt.length):
        for i in xrange(opt.out_size_mdim):
            for j in xrange(opt.out_size_smdim):
                data = "%f\t" % targets[0, step, i, j]
                f.write(data)
                data3 = "%f\t" % outputs[0, step, i, j]
                f3.write(data3)
        f.write("\n")
        f3.write("\n")

    f.close()
    f3.close()

# To plot the propOutput after the ERS
def display_er_output_prop(outputs, targets, opt):

    # For Prop. Output _ Open
    analog_pred = 0
    analog_target = 0
    for a in range(opt.out_size_smdim):
        analog_pred = analog_pred + outputs[0, :, :, a] * softmax_array[a]
        analog_target = analog_target + targets[0 , :, :, a] * softmax_array[a]

    #fig, ax = plt.subplots(nrows=2, sharex=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    analog_dim = targets.shape[2]
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, analog_dim)]
    for j in range(analog_dim):
        ax1.plot(analog_pred[0:-2, j], color=colors[j], label='prediction')
        ax1.plot(analog_target[1:-1, j], color=colors[j], linestyle='--', label='target')
        ax1.set_xlabel('Error Regression Closed-loop Prop.')

    plt.show()



# To save the visionOutput after the ERS
def save_er_output_vision(outputs, targets, opt):
    out_size_vrow = opt.out_size_vrow
    out_size_vcol = opt.out_size_vcol

    isdir = os.path.exists(opt.log_dir + "errorRegression")
    if not isdir:
        os.makedirs(opt.log_dir + "errorRegression")


    # Save Image
    f = open(opt.log_dir + "errorRegression" + "/target_vision_er.txt", 'w')
    f3 = open(opt.log_dir + "errorRegression" + "/closed_errorRegression_vision.txt", 'w')
    for step in xrange(opt.length):
        for height in xrange(out_size_vrow):
            for width in xrange(out_size_vcol):
                data = "%f\t" % targets[0, step, height, width]
                f.write(data)
                data3 = "%f\t" % outputs[0, step, height, width]
                f3.write(data3)
        f.write("\n")
        f3.write("\n")

    f.close()
    f3.close()


# To plot the visionOutput after the ERS
def display_er_output_vision(outputs, targets, opt):
    out_size_vrow = opt.out_size_vrow
    out_size_vcol = opt.out_size_vcol

    # setup figure
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.set_cmap('gray')

    ax1.set_xlabel('Target')
    ax2.set_xlabel('Error Regression')

    # set up list of images for animation
    ims = []
    for time in xrange(opt.length):
        im = ax1.imshow(targets[0,time, :, :])
        im2 = ax2.imshow(outputs[0,time, :, :])
        ims.append([im, im2])

    # run animation
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat=True)
    plt.show()



# After the ERS, this function is called and logs context activation
def save_er_logs_ctx(log_p1, log_p2, log_p3, log_v1, log_v2, log_v3, opt):
    print "=" * 100
    print "Saving the Ctx Logs..."

    length = opt.length

    fid_c_p1 = open(opt.log_dir + "errorRegression" + "/closed_p1.txt", 'w')
    fid_c_p2 = open(opt.log_dir + "errorRegression" + "/closed_p2.txt", 'w')
    fid_c_p3 = open(opt.log_dir + "errorRegression" + "/closed_p3.txt", 'w')
    fid_c_v1 = open(opt.log_dir + "errorRegression" + "/closed_v1.txt", 'w')
    fid_c_v2 = open(opt.log_dir + "errorRegression" + "/closed_v2.txt", 'w')
    fid_c_v3 = open(opt.log_dir + "errorRegression" + "/closed_v3.txt", 'w')


    iteration = -1
    for step in xrange(length):
        for fm in xrange(opt.p1_unit):
            data = "%f\t" % log_p1[step, fm, iteration]
            fid_c_p1.write(data)
        fid_c_p1.write("\n")

        for fm in xrange(opt.p2_unit):
            data = "%f\t" % log_p2[step, fm, iteration]
            fid_c_p2.write(data)
        fid_c_p2.write("\n")

        for fm in xrange(opt.p3_unit):
            for height in xrange(opt.p3_msize[0]):
                for width in xrange(opt.p3_msize[1]):
                    data = "%f\t" % log_p3[step, height, width, fm, iteration]
                    fid_c_p3.write(data)
        fid_c_p3.write("\n")

        for fm in xrange(opt.v1_unit):
            for height in xrange(opt.v1_msize[0]):
                for width in xrange(opt.v1_msize[1]):
                    data = "%f\t" % log_v1[step, height, width, fm, iteration]
                    fid_c_v1.write(data)
        fid_c_v1.write("\n")

        for fm in xrange(opt.v2_unit):
            for height in xrange(opt.v2_msize[0]):
                for width in xrange(opt.v2_msize[1]):
                    data = "%f\t" % log_v2[step, height, width, fm, iteration]
                    fid_c_v2.write(data)
        fid_c_v2.write("\n")

        for fm in xrange(opt.v3_unit):
            for height in xrange(opt.v3_msize[0]):
                for width in xrange(opt.v3_msize[1]):
                    data = "%f\t" % log_v3[step, height, width, fm, iteration]
                    fid_c_v3.write(data)
        fid_c_v3.write("\n")

    fid_c_p1.close()
    fid_c_p2.close()
    fid_c_v1.close()
    fid_c_v2.close()
    fid_c_v3.close()
    fid_c_p3.close()


# The initial value of the time window at each time step
def save_er_logs_ctx_initWind(log_p1, log_p2, log_p3, log_v1, log_v2, log_v3, opt):
    print "=" * 100
    print "Saving the Ctx.Init Logs in the window..."


    length = opt.length


    fid_c_p1 = open(opt.log_dir + "errorRegression" + "/closed_p1_initWind.txt", 'w')
    fid_c_p2 = open(opt.log_dir + "errorRegression" + "/closed_p2_initWind.txt", 'w')
    fid_c_p3 = open(opt.log_dir + "errorRegression" + "/closed_p3_initWind.txt", 'w')
    fid_c_v1 = open(opt.log_dir + "errorRegression" + "/closed_v1_initWind.txt", 'w')
    fid_c_v2 = open(opt.log_dir + "errorRegression" + "/closed_v2_initWind.txt", 'w')
    fid_c_v3 = open(opt.log_dir + "errorRegression" + "/closed_v3_initWind.txt", 'w')


    iteration = -1
    for step in xrange(length):
        for fm in xrange(opt.p1_unit):
            data = "%f\t" % log_p1[step, fm, iteration]
            fid_c_p1.write(data)
        fid_c_p1.write("\n")

        for fm in xrange(opt.p2_unit):
            data = "%f\t" % log_p2[step, fm, iteration]
            fid_c_p2.write(data)
        fid_c_p2.write("\n")

        for fm in xrange(opt.p3_unit):
            for height in xrange(opt.p3_msize[0]):
                for width in xrange(opt.p3_msize[1]):
                    data = "%f\t" % log_p3[step, height, width, fm, iteration]
                    fid_c_p3.write(data)
        fid_c_p3.write("\n")

        for fm in xrange(opt.v1_unit):
            for height in xrange(opt.v1_msize[0]):
                for width in xrange(opt.v1_msize[1]):
                    data = "%f\t" % log_v1[step, height, width, fm, iteration]
                    fid_c_v1.write(data)
        fid_c_v1.write("\n")

        for fm in xrange(opt.v2_unit):
            for height in xrange(opt.v2_msize[0]):
                for width in xrange(opt.v2_msize[1]):
                    data = "%f\t" % log_v2[step, height, width, fm, iteration]
                    fid_c_v2.write(data)
        fid_c_v2.write("\n")

        for fm in xrange(opt.v3_unit):
            for height in xrange(opt.v3_msize[0]):
                for width in xrange(opt.v3_msize[1]):
                    data = "%f\t" % log_v3[step, height, width, fm, iteration]
                    fid_c_v3.write(data)
        fid_c_v3.write("\n")

    fid_c_p1.close()
    fid_c_p2.close()
    fid_c_p3.close()
    fid_c_v1.close()
    fid_c_v2.close()
    fid_c_v3.close()



