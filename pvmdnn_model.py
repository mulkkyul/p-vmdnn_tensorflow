import tensorflow as tf
import functools
import basicPVMDNN as pvmdnn


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class Model(object):

    def __init__(self, motor, vision, idxd, cl_ratio, learning_rate, opt, rnn_cell = None, state_is_tuple=True):
        print('=' * 100)
        print('@ __init__')

        # General Hyper-parameters
        self._eps = 1e-8
        self._state_is_tuple = state_is_tuple
        self._idxd = idxd
        self._isThisTrain = opt.isThisTrain

        self._cl = cl_ratio # closed loop ratio
        self._lr = learning_rate # learning rate
        self._batch_size = opt.batch_size # batch size

        self._optimizer_name = opt.optimizer
        self._lossType = opt.lossType


        self._enableLateral_high_propToVision = opt.enableLateral_high_propToVision
        self._enableLateral_high_visionToProp = opt.enableLateral_high_visionToProp
        self._enableLateral_mid_propToVision = opt.enableLateral_mid_propToVision
        self._enableLateral_mid_visionToProp = opt.enableLateral_mid_visionToProp
        self._enableLateral_low_propToVision = opt.enableLateral_low_propToVision
        self._enableLateral_low_visionToProp = opt.enableLateral_low_visionToProp



        ### Hyper-parameters about the network configuration
        ## Proprioceptive Pathway
        # Output Layer
        self._out_size_smdim = opt.out_size_smdim  # softmax dim. of motor (= Number of SM units / analog dim)
        self._out_size_mdim = opt.out_size_mdim  # number of analog dim (= number of joints)
        self._out_size_prop = self._out_size_smdim * self._out_size_mdim
        self._pOut_unit = self._out_size_prop
        # Prop. Fast (lower level)
        self._p1_unit = opt.p1_unit
        self._p1_tau = opt.p1_tau
        # Prop. Middle (mid level)
        self._p2_unit = opt.p2_unit
        self._p2_tau = opt.p2_tau
        # Prop. Slow (higher level)
        # Note that Prop. Slow is handled as P-MSTRNN layer with 1 x 1 feature map 
        self._p3_unit = opt.p3_unit
        self._p3_msize = opt.p3_msize
        self._p3_size = opt.p3_size
        self._p3_tau = opt.p3_tau


        ## Visual Pathway
        # Output Layer
        self._out_size_vrow = opt.out_size_vrow
        self._out_size_vcol = opt.out_size_vcol
        self._out_size_vision = self._out_size_vrow*self._out_size_vcol
        # Vision Fast (lower level)
        self._v1_unit = opt.v1_unit                    # Number of Units
        self._v1_msize = opt.v1_msize                  # Map Size (2D)
        self._v1_size = opt.v1_size                    # Map Size (1D)

        self._v1_convFilter_h = opt.v1_convFilter_h
        self._v1_convFilter_w = opt.v1_convFilter_w
        self._v1_convStride_h = opt.v1_convStride_h
        self._v1_convStride_w = opt.v1_convStride_w

        self._v1_transConvFilter_h = opt.v2_convFilter_h
        self._v1_transConvFilter_w = opt.v2_convFilter_w
        self._v1_transConvStride_h = opt.v2_convStride_h
        self._v1_transConvStride_w = opt.v2_convStride_w

        self._v1_tau = opt.v1_tau
        # Vision Middle (mid level)
        self._v2_unit = opt.v2_unit
        self._v2_msize = opt.v2_msize
        self._v2_size = opt.v2_size

        self._v2_convFilter_h = opt.v2_convFilter_h
        self._v2_convFilter_w = opt.v2_convFilter_w
        self._v2_convStride_h = opt.v2_convStride_h
        self._v2_convStride_w = opt.v2_convStride_w

        self._v2_transConvFilter_h = opt.v3_convFilter_h
        self._v2_transConvFilter_w = opt.v3_convFilter_w
        self._v2_transConvStride_h = opt.v3_convStride_h
        self._v2_transConvStride_w = opt.v3_convStride_w
        self._v2_tau = opt.v2_tau

        # Vision Slow (higher level)
        self._v3_unit = opt.v3_unit
        self._v3_msize = opt.v3_msize
        self._v3_size = opt.v3_size

        self._v3_convFilter_h = opt.v3_convFilter_h
        self._v3_convFilter_w = opt.v3_convFilter_w
        self._v3_convStride_h = opt.v3_convStride_h
        self._v3_convStride_w = opt.v3_convStride_w

        self._v3_transConvFilter_h = opt.p3_convFilter_h
        self._v3_transConvFilter_w = opt.p3_convFilter_w
        self._v3_transConvStride_h = opt.p3_convStride_h
        self._v3_transConvStride_w = opt.p3_convStride_w

        self._v3_tau = opt.v3_tau


        self._p3_convFilter_h = opt.p3_convFilter_h
        self._p3_convFilter_w = opt.p3_convFilter_w
        self._p3_convStride_h = opt.p3_convStride_h
        self._p3_convStride_w = opt.p3_convStride_w


        # Input Nodes
        self._prop_in = motor[:,:-1,:,:] #motor input [batch, time, dim]
        self._prop_out = motor[:, 1:, :, :] #motor output
        self._prop_init = motor[:, 0, :]  # initial motor input for closed loop generation

        self._v_in = vision[:, :-1, :, :]  # vision input [batch,time,vrow,vcol]
        self._v_out = vision[:, 1:, :, :]  # vision output, 1-step prediction
        self._v_init = vision[:, 0, :, :]  # initial vision input for closed loop generation

        # Input Nodes for Testing (sensory entrainment & error regression)
        # Motor
        self._windInput_prop = tf.placeholder(tf.float32, shape=[None, None, self._out_size_mdim, self._out_size_smdim],
                                       name='testing_input_prop')
        self._windTarget_prop =  tf.placeholder(tf.float32, shape=[None, None, self._out_size_mdim, self._out_size_smdim],
                                       name='testing_target_prop')
        self._windInit_prop = tf.placeholder(tf.float32, shape=[None, self._out_size_mdim, self._out_size_smdim],
                                       name='testing_init_prop')

        # Vision input [batch,time,vrow,vcol]
        self._windInput_vision = tf.placeholder(tf.float32, shape=[None, None, self._out_size_vrow, self._out_size_vcol],
                                       name='testing_input_vision')
        self._windTarget_vision = tf.placeholder(tf.float32, shape=[None, None, self._out_size_vrow, self._out_size_vcol],
                                       name='testing_target_vision')
        self._windInit_vision = tf.placeholder(tf.float32, shape=[None, self._out_size_vrow, self._out_size_vcol],
                                       name='testing_init_vision')


        # Initial States
        # h: initial states (internal states), y: neuron activation (after activation function)

        # Prop. Fast
        self._myInit_p1_h = tf.get_variable('init_p1_h', shape=[opt.num_data, self._p1_unit],
                                           initializer=tf.constant_initializer(0.0), trainable=True)
        self._myInit_p1_y = pvmdnn.tanh_mod(self._myInit_p1_h)

        # Prop. Mid
        self._myInit_p2_h = tf.get_variable('init_p2_h', shape=[opt.num_data, self._p2_unit],
                                           initializer=tf.constant_initializer(0.0), trainable=True)
        self._myInit_p2_y = pvmdnn.tanh_mod(self._myInit_p2_h)
        # Prop. Slow
        self._myInit_p3_h = tf.get_variable('init_p3_h',
                                            shape=[opt.num_data, self._p3_msize[0], self._p3_msize[1], self._p3_unit],
                                            initializer=tf.constant_initializer(0.0), trainable=True)
        self._myInit_p3_y = pvmdnn.tanh_mod(self._myInit_p3_h)

        # Vision Fast
        self._myInit_v1_h = tf.get_variable('init_v1_h',
                                            shape=[opt.num_data, self._v1_msize[0], self._v1_msize[1], self._v1_unit],
                                            initializer=tf.constant_initializer(0.0), trainable=True)
        self._myInit_v1_y = pvmdnn.tanh_mod(self._myInit_v1_h)
        # Vision Mid
        self._myInit_v2_h = tf.get_variable('init_v2_h',
                                            shape=[opt.num_data, self._v2_msize[0], self._v2_msize[1], self._v2_unit],
                                            initializer=tf.constant_initializer(0.0), trainable=True)
        self._myInit_v2_y = pvmdnn.tanh_mod(self._myInit_v2_h)
        # Vision Slow
        self._myInit_v3_h = tf.get_variable('init_v3_h',
                                            shape=[opt.num_data, self._v3_msize[0], self._v3_msize[1], self._v3_unit],
                                            initializer=tf.constant_initializer(0.0), trainable=True)
        self._myInit_v3_y = pvmdnn.tanh_mod(self._myInit_v3_h)


        # Initial States for testing
        self._windInit_p1_h = tf.placeholder(tf.float32, shape=[opt.num_data, self._p1_unit], name='wind_init_p1_h')
        self._windInit_p2_h = tf.placeholder(tf.float32, shape=[opt.num_data, self._p2_unit], name='wind_init_p2_h')
        self._windInit_p3_h = tf.placeholder(tf.float32, shape=[opt.num_data, self._p3_msize[0], self._p3_msize[1], self._p3_unit],
                                            name='wind_init_p3_h')


        self._windInit_v1_h = tf.placeholder(tf.float32, shape=[opt.num_data, self._v1_msize[0], self._v1_msize[1], self._v1_unit],
                                            name='wind_init_v1_h')
        self._windInit_v2_h = tf.placeholder(tf.float32, shape=[opt.num_data, self._v2_msize[0], self._v2_msize[1], self._v2_unit],
                                            name='wind_init_v2_h')
        self._windInit_v3_h = tf.placeholder(tf.float32, shape=[opt.num_data, self._v3_msize[0], self._v3_msize[1], self._v3_unit],
                                            name='wind_init_v3_h')


        ## Setting the layers in the proprioceptive & visual pathways

        # Prop. Fast (lower level)
        with tf.variable_scope('p1'):
            self.cell_p1 = pvmdnn.BasicCTRNNCell(self._p1_unit, state_is_tuple=self._state_is_tuple, tau=self._p1_tau)
        # Prop. Middle (mid level)
        with tf.variable_scope('p2'):
            self.cell_p2 = pvmdnn.BasicCTRNNCell(self._p2_unit, state_is_tuple=self._state_is_tuple, tau=self._p2_tau)
        # Prop. Slow (higher level)
        # Note that this Prop. Slow is handled as PMSTRNN layer with a size of 1 x 1 feature map.
        with tf.variable_scope('p3'):
            self.cell_p3 = pvmdnn.BasicPMSTRNNCell(self._p3_size, self._p3_msize, self._p3_unit, state_is_tuple=self._state_is_tuple, tau=self._p3_tau)

        # Vision Fast (lower level)
        with tf.variable_scope('v1'):
            self.cell_v1 = pvmdnn.BasicPMSTRNNCell(self._v1_size, self._v1_msize, self._v1_unit, state_is_tuple=self._state_is_tuple, tau=self._v1_tau)
        # Vision Middle (mid level)
        with tf.variable_scope('v2'):
            self.cell_v2 = pvmdnn.BasicPMSTRNNCell(self._v2_size, self._v2_msize, self._v2_unit, state_is_tuple=self._state_is_tuple, tau=self._v2_tau)
        # Vision Slow (higher level)
        with tf.variable_scope('v3'):
            self.cell_v3 = pvmdnn.BasicPMSTRNNCell(self._v3_size, self._v3_msize, self._v3_unit, state_is_tuple=self._state_is_tuple, tau=self._v3_tau)


        ## Setting up the variables (weights & biases) for output layers in each pathway

        # Weights from Prop. Fast to Prop. Out
        self.w_prop_out = tf.get_variable('w_prop_out', shape=[self._p1_unit, self._out_size_prop])
        # Biases
        self.b_prop_out = tf.get_variable('b_prop_out', shape=[self._out_size_prop], initializer=tf.constant_initializer(0.0))

        # Weights from Vision Fast to Vision Out
        # Deprecated (Replaced by _conv_linear_pmstrnn_fromPadded)
        #self.w_vision_out = tf.get_variable('w_vision_out', shape=[5, 5, 1, self._v1_unit])
        # Biases
        #self.b_vision_out = tf.get_variable('b_vision_out', shape=[self._out_size_vision], initializer=tf.constant_initializer(0.0))


        # graphs
        self.prediction_pmstrnn
        self.optimize
        self.optimize_testing

    def model_step_pmstrnn(self, model_input, model_out_prev):
        print('=' * 100)
        print('@ model_step_pmstrnn')
        input_prop, input_vision = model_input
        prev_prop, prev_cell_p1, prev_cell_p2, prev_cell_p3, \
        prev_vision, prev_cell_v1, prev_cell_v2, prev_cell_v3  = model_out_prev

        input_vision = tf.reshape(input_vision, [-1, self._out_size_vrow, self._out_size_vcol, 1])
        prev_vision = tf.reshape(prev_vision, [-1, self._out_size_vrow, self._out_size_vcol, 1])

        # Separate all cell states
        prev_out_p1, prev_state_p1 = prev_cell_p1
        prev_out_p2, prev_state_p2 = prev_cell_p2
        prev_out_p3, prev_state_p3 = prev_cell_p3
        prev_out_v1, prev_state_v1 = prev_cell_v1
        prev_out_v2, prev_state_v2 = prev_cell_v2
        prev_out_v3, prev_state_v3 = prev_cell_v3

        # Setting up the open-loop and closed-loop ratio
        cur_prop = tf.multiply(prev_prop, self._cl) + tf.multiply(input_prop, 1 - self._cl)
        cur_vision = tf.multiply(prev_vision, self._cl) + tf.multiply(input_vision, 1 - self._cl)


        print "=" * 100
        print "Initialize Proprioception Fast (Lower Level)"
        input_p1 = tf.concat(axis=1, values=[tf.reshape(cur_prop, [-1, self._out_size_prop]),
                                 tf.reshape(prev_out_p2, [-1, self._p2_unit])])
        if self._enableLateral_low_visionToProp:
            input_p1_conv = prev_out_v1
            convProp_strides = [1, 1, 1, 1]
            convProp_filter = [self._v1_msize[0], self._v1_msize[1]]
            cell_p1 = self.cell_p1.feed_input(input_p1, prev_cell_p1, input_p1_conv, convProp_strides, convProp_filter, scope = 'p1')
        else:
            cell_p1 = self.cell_p1(input_p1, prev_cell_p1, scope='p1')

        cell_out_p1, cell_state_p1 = cell_p1

        print "=" * 100
        print "Initialize Proprioception Mid (Mid Level)"
        input_p2 = tf.concat(axis=1, values=[tf.reshape(prev_out_p1, [-1, self._p1_unit]),
                                 tf.reshape(prev_out_p3, [-1, self._p3_unit])])
        if self._enableLateral_mid_visionToProp:
            input_m2_conv = prev_out_v2
            convProp_strides = [1, 1, 1, 1]
            convProp_filter = [self._v2_msize[0], self._v2_msize[1]]
            cell_p2 = self.cell_p2.feed_input(input_p2, prev_cell_p2, input_m2_conv, convProp_strides,
                                              convProp_filter, scope='p2')
        else:
            cell_p2 = self.cell_p2(input_p2, prev_cell_p2, scope='p2')

        print "=" * 100
        print "Initialize Proprioception Slow (Higher Level)"
        input_p3_conv = prev_out_v3
        input_p3_fromProp = tf.reshape(prev_out_p2, [-1, 1, 1, self._p2_unit])
        conv_strides = [1, self._p3_convStride_h, self._p3_convStride_w, 1]
        conv_filter = [self._p3_convFilter_h, self._p3_convFilter_w]
        if self._enableLateral_high_visionToProp:
            cell_p3 = self.cell_p3.feed_input_propTop(input_p3_conv, input_p3_fromProp, prev_cell_p3, conv_strides, conv_filter, scope='p3')
        else:
            cell_p3 = self.cell_p3.feed_input_propTop_noLateral(input_p3_fromProp, prev_cell_p3, scope='p3')



        print "=" * 100
        print "Initialize Vision Fast (Lower Level)"
        input_v1_conv = cur_vision
        conv_strides = [1, self._v1_convStride_h, self._v1_convStride_w, 1]
        conv_filter = [self._v1_convFilter_h, self._v1_convFilter_w]
        input_v1_trconv = prev_out_v2
        trconv_strides = [1, self._v1_transConvStride_h, self._v1_transConvStride_w, 1]
        trconv_filter = [self._v1_transConvFilter_h, self._v1_transConvFilter_w]

        if self._enableLateral_mid_propToVision:
            input_f1_motor = tf.reshape(prev_out_p1, [-1, 1, 1, self._p1_unit])
            trconvMotor_strides = [1, 1, 1, 1]
            trconvMotor_filter = [self._v1_msize[0], self._v1_msize[1]]
            cell_v1 = self.cell_v1.feed_input_withProp(input_v1_conv, input_v1_trconv, prev_cell_v1,
                                                    conv_strides, conv_filter, trconv_strides, trconv_filter,
                                                    input_f1_motor, trconvMotor_strides, trconvMotor_filter, scope='v1')
        else:
            cell_v1 = self.cell_v1.feed_input(input_v1_conv, input_v1_trconv, prev_cell_v1,
                                              conv_strides, conv_filter, trconv_strides, trconv_filter, scope='v1')

        cell_out_v1, cell_state_v1 = cell_v1


        print "=" * 100
        print "Initialize Vision Middle (Mid Level)"
        input_v2_conv = prev_out_v1
        conv_strides = [1, self._v2_convStride_h, self._v2_convStride_w, 1]
        conv_filter = [self._v2_convFilter_h, self._v2_convFilter_w]
        input_v2_trconv = prev_out_v3
        trconv_strides = [1, self._v2_transConvStride_h, self._v2_transConvStride_w, 1]
        trconv_filter = [self._v2_transConvFilter_h, self._v2_transConvFilter_w]

        if self._enableLateral_mid_propToVision:
            input_f2_motor = tf.reshape(prev_out_p2, [-1, 1, 1, self._p2_unit])
            trconvMotor_strides = [1, 1, 1, 1]
            trconvMotor_filter = [self._v2_msize[0], self._v2_msize[1]]
            cell_v2 = self.cell_v2.feed_input_withProp(input_v2_conv, input_v2_trconv, prev_cell_v2,
                                                    conv_strides, conv_filter, trconv_strides, trconv_filter,
                                                    input_f2_motor, trconvMotor_strides, trconvMotor_filter, scope='v2')
        else:
            cell_v2 = self.cell_v2.feed_input(input_v2_conv, input_v2_trconv, prev_cell_v2,
                                              conv_strides, conv_filter, trconv_strides, trconv_filter, scope='v2')


        print "=" * 100
        print "Initialize Vision Slow (Higher Level)"
        input_v3_conv = prev_out_v2
        conv_strides = [1, self._v3_convStride_h, self._v3_convStride_w, 1]
        conv_filter = [self._v3_convFilter_h, self._v3_convFilter_w]
        input_v3_trconv = prev_out_p3
        trconv_strides = [1, self._v3_transConvStride_h, self._v3_transConvStride_w, 1]
        trconv_filter = [self._v3_transConvFilter_h, self._v3_transConvFilter_w]
        if self._enableLateral_high_visionToProp:
            cell_v3 = self.cell_v3.feed_input(input_v3_conv, input_v3_trconv, prev_cell_v3,
                                              conv_strides, conv_filter, trconv_strides, trconv_filter, scope='v3')
        else:
            cell_v3 = self.cell_v3.feed_input_top_noPropInput(input_v3_conv, prev_cell_v3,
                                              conv_strides, conv_filter, scope='v3')


        # Proprioceptive Output
        logit_motor = tf.matmul(cell_out_p1,self.w_prop_out) + self.b_prop_out
        logit_motor_rs = tf.reshape(logit_motor,[-1,self._out_size_mdim, self._out_size_smdim]) # reshape for softmax
        pred_step_prop = tf.nn.softmax(logit_motor_rs)  # softmax connot be at the outside b.o. closed loop (it might be faster...)

        # Vision Output (With Padding)
        padding_height = (self._out_size_vrow - self._v1_msize[0]) / 2
        padding_width = (self._out_size_vcol - self._v1_msize[1]) / 2
        paddings = [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]]
        cell_out_v1_padded = tf.pad(cell_out_v1, paddings, mode='CONSTANT', name=None)
        out_u = pvmdnn._conv_linear_pmstrnn_fromPadded([cell_out_v1_padded], [self._v1_convFilter_h, self._v1_convFilter_w], 1, bias=True, paddingType="SAME")
        logit_vision = pvmdnn.tanh_mod(out_u)
        pred_step_vision = tf.reshape(logit_vision, [-1, self._out_size_vrow, self._out_size_vcol])
        
        # Model Output (including each layer's state)
        model_out = (
            pred_step_prop, cell_p1[1], cell_p2[1], cell_p3[1],
            pred_step_vision, cell_v1[1], cell_v2[1], cell_v3[1])

        try:
            input("Press enter to continue")
        except SyntaxError:
            pass


        return model_out


    # Forward Dynamics - Obtain both visual & prop. predictions
    @lazy_property
    def prediction_pmstrnn(self):
        print('=' * 100)
        print('@ prediction _pmstrnn')

        if self._isThisTrain:
            # transpose inputs for scan and make tuple
            v_t = tf.transpose(self._v_in, perm=[1, 0, 2, 3])
            m_t = tf.transpose(self._prop_in, perm=[1, 0, 2, 3])
            input_t = (m_t, v_t)
        else:
            # transpose inputs for scan and make tuple
            v_t = tf.transpose(self._windInput_vision, perm=[1, 0, 2, 3])
            m_t = tf.transpose(self._windInput_prop, perm=[1, 0, 2, 3])
            input_t = (m_t, v_t)
            #dyn_input_shape = tf.shape(self._windInput_vision)
            #batch_size = dyn_input_shape[0]


        # make the initializer for the scan function
        # Using the embedding_lookup, it reads the corresponding initial states from the variables
        # h: internal states, y: activation value

        # Prop. Fast
        myInit_p1_h = tf.nn.embedding_lookup(self._myInit_p1_h, self._idxd)
        myInit_p1_h = tf.reshape(myInit_p1_h, [-1, self._p1_unit])
        myInit_p1_y = tf.nn.embedding_lookup(self._myInit_p1_y, self._idxd)
        myInit_p1_y = tf.reshape(myInit_p1_y, [-1, self._p1_unit])
        new_c_p1 = tf.nn.rnn_cell.LSTMStateTuple(myInit_p1_y, myInit_p1_h)
        # Prop. Mid
        myInit_p2_h = tf.nn.embedding_lookup(self._myInit_p2_h, self._idxd)
        myInit_p2_h = tf.reshape(myInit_p2_h, [-1, self._p2_unit])
        myInit_p2_y = tf.nn.embedding_lookup(self._myInit_p2_y, self._idxd)
        myInit_p2_y = tf.reshape(myInit_p2_y, [-1, self._p2_unit])
        new_c_p2 = tf.nn.rnn_cell.LSTMStateTuple(myInit_p2_y, myInit_p2_h)
        # Prop. Slow
        myInit_p3_h = tf.nn.embedding_lookup(self._myInit_p3_h, self._idxd)
        myInit_p3_y = tf.nn.embedding_lookup(self._myInit_p3_y, self._idxd)
        new_c_p3 = tf.nn.rnn_cell.LSTMStateTuple(myInit_p3_y, myInit_p3_h)
        
        # Vision Fast
        myInit_v1_h = tf.nn.embedding_lookup(self._myInit_v1_h, self._idxd)
        myInit_v1_y = tf.nn.embedding_lookup(self._myInit_v1_y, self._idxd)
        new_c_v1 = tf.nn.rnn_cell.LSTMStateTuple(myInit_v1_y, myInit_v1_h)
        # Vision Mid
        myInit_v2_h = tf.nn.embedding_lookup(self._myInit_v2_h, self._idxd)
        myInit_v2_y = tf.nn.embedding_lookup(self._myInit_v2_y, self._idxd)
        new_c_v2 = tf.nn.rnn_cell.LSTMStateTuple(myInit_v2_y, myInit_v2_h)
        # Vision Slow
        myInit_v3_h = tf.nn.embedding_lookup(self._myInit_v3_h, self._idxd)
        myInit_v3_y = tf.nn.embedding_lookup(self._myInit_v3_y, self._idxd)
        new_c_v3 = tf.nn.rnn_cell.LSTMStateTuple(myInit_v3_y, myInit_v3_h)

        if self._isThisTrain:
            init_state = (
                self._prop_init, new_c_p1, new_c_p2, new_c_p3,
                self._v_init, new_c_v1, new_c_v2, new_c_v3)
        else:
            init_state = (
                self._windInit_prop, new_c_p1, new_c_p2, new_c_p3,
                self._windInit_vision, new_c_v1, new_c_v2, new_c_v3)

        scan_outputs = tf.scan(lambda a, x: self.model_step_pmstrnn(x, a), input_t, initializer=init_state)

        pred_prop_t, c_p1_t, c_p2_t, c_p3_t, \
        pred_vision_t, c_v1_t, c_v2_t, c_v3_t = scan_outputs


        pred_prop = tf.transpose(pred_prop_t, perm=[1, 0, 2, 3], name='pred_prop')
        c_p1 = tf.transpose(c_p1_t, perm=[2, 1, 0, 3], name='states_propFast')
        c_p2 = tf.transpose(c_p2_t, perm=[2, 1, 0, 3], name='states_propMid')
        c_p3 = tf.transpose(c_p3_t, perm=[2, 1, 0, 3, 4, 5], name='states_propSlow')

        pred_vision = tf.transpose(pred_vision_t, perm=[1, 0, 2, 3], name='pred_vision')
        c_v1 = tf.transpose(c_v1_t, perm=[2, 1, 0, 3, 4, 5], name='states_visionFast')
        c_v2 = tf.transpose(c_v2_t, perm=[2, 1, 0, 3, 4, 5], name='states_visionMid')
        c_v3 = tf.transpose(c_v3_t, perm=[2, 1, 0, 3, 4, 5], name='states_visionSlow')

        return pred_prop, c_p1, c_p2, c_p3, pred_vision, c_v1, c_v2, c_v3, input_t, init_state, self._idxd



    # To calculate loss
    @lazy_property
    def cost(self):
        # Obtain proprioceptive prediction (pred_prop) & visual prediction (pred_vision)
        pred_prop, _, _, _, pred_vision, _, _, _, _,_, _ = self.prediction_pmstrnn

        # Calculate loss in both visual & proprioceptive pathway
        loss_vision = tf.reduce_mean((self._v_out - pred_vision) ** 2, name='loss_vision')
        loss_prop = tf.reduce_mean(
            -tf.reduce_sum(self._prop_out * (tf.log(pred_prop + self._eps) - tf.log(self._prop_out + self._eps)),
                           axis=[2, 3]), name='loss_proprioception')

        # Total loss = sum of both losses
        loss = loss_vision + loss_prop

        return loss, loss_prop, loss_vision


    # To optimize the model's learnable parameters (variables)
    @lazy_property
    def optimize(self):
        # Obtain the loss from self.cost
        loss, _, _ = self.cost

        # Choose the optimizer (specified in setting.ini)
        if self._optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(self._lr)
        elif self._optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        else:
            optimizer = tf.train.AdamOptimizer(self._lr)

        # optimize the entire variables
        return optimizer.minimize(loss)

    # To get the initial states of the model (this is used for checking the training result)
    @lazy_property
    def get_initStates(self):
        return self._myInit_p1_h, self._myInit_p2_h, self._myInit_p3_h, self._myInit_v1_h, self._myInit_v2_h, self._myInit_v3_h

    #=============================================================================
    # For Testing (sensory entrainment & error regression)
    # =============================================================================

    @lazy_property
    def cost_testing(self):
        pred_p, _, _, _, pred_v, _, _, _, input_t, init_state, iidx = self.prediction_pmstrnn
        loss_v = tf.reduce_mean((self._windTarget_vision - pred_v) ** 2, name='loss_vision')
        loss_p = tf.reduce_mean(
            -tf.reduce_sum(self._windTarget_prop * (tf.log(pred_p + self._eps) - tf.log(self._windTarget_prop + self._eps)),
                           axis=[2, 3]), name='loss_motor')
        if self._lossType == 1:
            loss = loss_v
        elif self._lossType == 2:
            loss = loss_p
        elif self._lossType == 3:
            loss = loss_p + loss_v

        return loss, loss_v, loss_p

    @lazy_property
    def optimize_testing(self):
        loss, _, _ = self.cost_testing
        if self._optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(self._lr)
        elif self._optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        else:
            optimizer = tf.train.AdamOptimizer(self._lr)

        return optimizer.minimize(loss, var_list=[self._myInit_p1_h, self._myInit_p2_h, self._myInit_p3_h,
            self._myInit_v1_h, self._myInit_v2_h, self._myInit_v3_h])

    @lazy_property
    def set_wInitState(self):
        self._myInit_p1_h = tf.assign(self._myInit_p1_h, self._windInit_p1_h)
        self._myInit_p2_h = tf.assign(self._myInit_p2_h, self._windInit_p2_h)
        self._myInit_p3_h = tf.assign(self._myInit_p3_h, self._windInit_p3_h)
        self._myInit_v1_h = tf.assign(self._myInit_v1_h, self._windInit_v1_h)
        self._myInit_v2_h = tf.assign(self._myInit_v2_h, self._windInit_v2_h)
        self._myInit_v3_h = tf.assign(self._myInit_v3_h, self._windInit_v3_h)

        return self._myInit_p1_h, self._myInit_p2_h, self._myInit_p3_h, self._myInit_v1_h, self._myInit_v2_h, self._myInit_v3_h


    @property
    def get_pred_m(self):
        pred_prop, c_p1, c_p2, c_p3, pred_vision, c_v1, c_v2, c_v3, input_t, init_state, iidx = self.prediction_pmstrnn
        return pred_prop, c_p1, c_p2, c_p3, pred_vision, c_v1, c_v2, c_v3, input_t, init_state, iidx
