#=========================================================================#
# The basic functions for P-MSTRNN (Visual Pathway) and MTRNN (Proprioceptive Pathway)
# This code is written based on TensorFlow's RNN, LSTM  and ConvRNN Codes.
# Some common notations in the code
# u: internal states (the one before the activation function)
# h (or y): activation value (the one after applying the activation function on u)
# state: LSTMStateTuple(u, h) or LSTMStateTuple(u, y)
# tau: time constants
# eta: 1 / tau
#=========================================================================#

from __future__ import absolute_import
from __future__ import division

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

import tensorflow as tf


# =============================================================================
# The following activation function was recommended in:
# Y. A. LeCun, L. Bottou, G. B. Orr, and K.-R. Muller, "Efficient backprop," in Neural Networks: Tricks of the Trade. vol. 7700,
# G. Montavon, G. B. Orr, and K.-R. Muller, Eds., ed: Springer Berlin Heidelberg, 2012, pp. 9-48.
# =============================================================================
def tanh_mod(x):
    return 1.7159 * tf.tanh(0.66666667 * x)
# =============================================================================



# =============================================================================
# _linear functions that calculates sum(input * weights) + biases
# =============================================================================
## Used in MTRNN layers
def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(args, axis=1), matrix)
    if not bias:
      return res
    bias_term = vs.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=init_ops.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term
# =============================================================================
## Used in P-MSTRNN layers
# ConvLinear (the size of the input feature map is bigger than the size of the output feature map)
def _conv_linear_pmstrnn(args, filter_size, filter_strides, num_features, bias, paddingType, bias_start=0.0, scope=None):
    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
        print "W : ", matrix.get_shape()
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=filter_strides, padding=paddingType)
        else:
            res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, strides=filter_strides, padding='VALID')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term
# TransposedConvLinear (the size of the input feature map is smaller than the size of the output feature map)
def _conv_transpose_linear(args, filter_size, filter_strides, num_features, output_shape, bias, bias_start=0.0, scope=None):
    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]


    with tf.variable_scope(scope or "ConvTranspose"):
        # "Matrix", [filter_shape[0], filter_shape[1], output's # of FM, src's # of FM], dtype = dtype)
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], num_features, total_arg_size_depth], dtype=dtype)
        print "W : ", matrix.get_shape()

        dyn_input_shape = tf.shape(args[0])
        batch_size = dyn_input_shape[0]
        output_shape2 = tf.stack([batch_size, output_shape[0], output_shape[1], num_features])

        res= tf.nn.conv2d_transpose(args[0], matrix, output_shape=output_shape2,strides=filter_strides, padding='VALID')

        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term
def _conv_linear_pmstrnn_fromPadded(args, filter_size, num_features, bias, paddingType, bias_start=0.0, scope=None):
    """convolution:
    Args:
      args: a 4D Tensor or a list of 4D, batch x n, Tensors.
      filter_size: int tuple of filter height and width.
      num_features: int, number of features.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 4D Tensor with shape [batch h w num_features]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding=paddingType)
            #res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='VALID')
            #res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, strides=[1, 1, 1, 1], padding='VALID')
            #res = tf.nn.conv2d(tf.concat(3, args), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term
# =============================================================================



class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """
    def __call__(self, inputs, state, scope=None):
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """
        shape = self.shape
        num_features = self.num_features
        zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2])
        return zeros

# =============================================================================
# P-MSTRNN Layers (Visual Pathway)
# =============================================================================
class BasicPMSTRNNCell(ConvRNNCell):
    def __init__(self, shape, msize, num_features, input_size=None,
                 state_is_tuple=False, activation=tanh_mod, tau=1.0):
        self.shape = shape
        self.msize = msize
        self.num_features = num_features
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._tau = tau
        self._eta = 1.0 / self._tau

    @property
    def state_size(self):
        return (tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    # P-MSTRNN layer when there's no lateral connection from MTRNN layer (Proprioceptive Pathway)
    def feed_input(self, inputs_conv_valid, inputs_trconv_valid, state, conv_strides, conv_filter, trconv_strides, trconv_filter, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            prev_f_u, prev_f_h = state

            print "INPUT (CONV.VALID) : ", inputs_conv_valid.get_shape()
            print "FILTER SIZE, STRIDES : " , conv_filter, conv_strides
            f_u = _conv_linear_pmstrnn([inputs_conv_valid], conv_filter, conv_strides, self.num_features, bias=True, paddingType='VALID', scope='CONV_VALID')
            print "AFTER CONV : ", f_u.get_shape()

            print '-' * 90
            output_shape = [self.msize[0], self.msize[1]]
            print "INPUT (TR.CONV.VALID) : ", inputs_trconv_valid.get_shape()
            print "FILTER SIZE, STRIDES : ", trconv_filter, trconv_strides
            f_u2 = _conv_transpose_linear([inputs_trconv_valid], trconv_filter, trconv_strides, self.num_features, output_shape, bias=False)
            print "AFTER TR.CONV : ", f_u2.get_shape()

            print '-' * 90
            print "INPUT (CONV.VALID.RECURRNT) : ", prev_f_h.get_shape()
            rc_filter = [2,2]
            rc_strides = [1,1,1,1]
            print "FILTER SIZE, STRIDES : " , rc_filter, rc_strides
            f_u3 = _conv_linear_pmstrnn([prev_f_h], rc_filter, rc_strides, self.num_features, bias=False, paddingType='SAME', scope='RC')
            print "AFTER CONV : ", f_u.get_shape()

            print '-' * 90
            f_u = (1 - self._eta) * prev_f_u + self._eta * (f_u + f_u2 + f_u3)
            new_h = self._activation(f_u)
            print "OUTPUT_SHAPE : ", output_shape
            print "MAP SIZE : ", new_h.get_shape()

            new_state = tf.nn.rnn_cell.LSTMStateTuple(f_u, new_h)

            return new_h, new_state

    # P-MSTRNN layer when there's lateral connection from MTRNN layer (Proprioceptive Pathway)
    def feed_input_withProp(self, inputs_conv_valid, inputs_trconv_valid, state, conv_strides, conv_filter,
                         trconv_strides,
                         trconv_filter, input_f1_motor, trconvMotor_strides, trconvMotor_filter, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            if self._state_is_tuple:
                prev_f_u, prev_f_h = state
            else:
                prev_f_u, prev_f_h = tf.split(axis=3, num_or_size_splits=2, value=state)

            print "INPUT (CONV.VALID) : ", inputs_conv_valid.get_shape()
            print "FILTER SIZE, STRIDES : ", conv_filter, conv_strides
            f_u = _conv_linear_pmstrnn([inputs_conv_valid], conv_filter, conv_strides, self.num_features, bias=True,
                                       paddingType='VALID', scope='CONV_VALID')  # ??
            print "AFTER CONV : ", f_u.get_shape()

            print '-' * 90
            output_shape = [self.msize[0], self.msize[1]]
            print "INPUT (TR.CONV.VALID) : ", inputs_trconv_valid.get_shape()
            print "FILTER SIZE, STRIDES : ", trconv_filter, trconv_strides
            f_u2 = _conv_transpose_linear([inputs_trconv_valid], trconv_filter, trconv_strides, self.num_features,
                                          output_shape, bias=False)
            print "AFTER TR.CONV : ", f_u2.get_shape()

            print '-' * 90
            print "INPUT (CONV.VALID.RECURRNT) : ", prev_f_h.get_shape()
            rc_filter = [2, 2]
            rc_strides = [1, 1, 1, 1]
            print "FILTER SIZE, STRIDES : ", rc_filter, rc_strides
            f_u3 = _conv_linear_pmstrnn([prev_f_h], rc_filter, rc_strides, self.num_features, bias=False,
                                        paddingType='SAME', scope='RC')  # ??
            print "AFTER CONV : ", f_u.get_shape()

            print '-' * 90
            output_shape = [self.msize[0], self.msize[1]]
            print "INPUT (FROM.MOTOR) : ", input_f1_motor.get_shape()
            print "FILTER SIZE, STRIDES : ", trconvMotor_filter, trconvMotor_strides
            f_u4 = _conv_transpose_linear([input_f1_motor], trconvMotor_filter, trconvMotor_strides,
                                          self.num_features,
                                          output_shape, bias=False, scope='trconvMotor')
            print "AFTER TR.CONV : ", f_u4.get_shape()

            print '-' * 90
            f_u = (1 - self._eta) * prev_f_u + self._eta * (f_u + f_u2 + f_u3 + f_u4)
            new_h = self._activation(f_u)
            print "OUTPUT_SHAPE : ", output_shape
            print "MAP SIZE : ", new_h.get_shape()

            if self._state_is_tuple:
                new_state = tf.nn.rnn_cell.LSTMStateTuple(f_u, new_h)
            else:
                new_state = tf.concat(axis=3, values=[f_u, new_h])
            return new_h, new_state

    # P-MSTRNN layer used for Vision Slow (the highest layer in the visual pathway) only
    # (when there's no lateral connection from Proprioceptive Pathway)
    def feed_input_top_noPropInput(self, inputs_conv_valid, state, conv_strides, conv_filter, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            prev_f_u, prev_f_h = state

            print "INPUT (CONV.VALID) : ", inputs_conv_valid.get_shape()
            print "FILTER SIZE, STRIDES : ", conv_filter, conv_strides
            f_u = _conv_linear_pmstrnn([inputs_conv_valid], conv_filter, conv_strides, self.num_features, bias=True,
                                       paddingType='VALID', scope='CONV_VALID')
            print "AFTER CONV : ", f_u.get_shape()

            print '-' * 90
            print "INPUT (CONV.VALID.RECURRNT) : ", prev_f_h.get_shape()
            rc_filter = [2, 2]
            rc_strides = [1, 1, 1, 1]
            print "FILTER SIZE, STRIDES : ", rc_filter, rc_strides
            f_u3 = _conv_linear_pmstrnn([prev_f_h], rc_filter, rc_strides, self.num_features, bias=False,
                                        paddingType='SAME', scope='RC')
            print "AFTER CONV : ", f_u.get_shape()

            print '-' * 90
            f_u = (1 - self._eta) * prev_f_u + self._eta * (f_u + f_u3)
            new_h = self._activation(f_u)
            output_shape = [self.msize[0], self.msize[1]]
            print "OUTPUT_SHAPE : ", output_shape
            print "MAP SIZE : ", new_h.get_shape()

            new_state = tf.nn.rnn_cell.LSTMStateTuple(f_u, new_h)

            return new_h, new_state

    # This is a P-MSTRNN layer used for Prop. Slow layer only (the highest prop. layer (Prop. Slow)
    # Note that Prop. Slow layer is handled as P-MSTRNN layer with a size of 1 x 1 feature map
    # This is for the one with a lateral input (from visual pathway)
    def feed_input_propTop(self, inputs_conv_valid, inputs_motor, state, conv_strides, conv_filter, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            prev_f_u, prev_f_h = state

            print "INPUT (CONV.VALID) : ", inputs_conv_valid.get_shape()
            print "FILTER SIZE, STRIDES : ", conv_filter, conv_strides
            f_u = _conv_linear_pmstrnn([inputs_conv_valid], conv_filter, conv_strides, self.num_features, bias=True,
                                       paddingType='VALID', scope='CONV_VALID')  # ??
            print "AFTER CONV : ", f_u.get_shape()

            print '-' * 90
            print "INPUT (MOTOR) : ", inputs_motor.get_shape()
            mt_filter = [1, 1]
            mt_strides = [1, 1, 1, 1]
            print "FILTER SIZE, STRIDES : ", mt_filter, mt_strides
            f_u2 = _conv_linear_pmstrnn([inputs_motor], mt_filter, mt_strides, self.num_features, bias=False,
                                       paddingType='VALID', scope='MOTOR')  # ??
            print "AFTER CONV : ", f_u.get_shape()

            print '-' * 90
            print "INPUT (CONV.VALID.RECURRNT) : ", prev_f_h.get_shape()
            rc_filter = [1, 1]
            rc_strides = [1, 1, 1, 1]
            print "FILTER SIZE, STRIDES : ", rc_filter, rc_strides
            f_u3 = _conv_linear_pmstrnn([prev_f_h], rc_filter, rc_strides, self.num_features, bias=False,
                                        paddingType='SAME', scope='RC')  # ??
            print "AFTER CONV : ", f_u.get_shape()

            print '-' * 90
            f_u = (1 - self._eta) * prev_f_u + self._eta * (f_u + f_u2 + f_u3)
            new_h = self._activation(f_u)
            print "OUTPUT_SHAPE : ", rc_filter
            print "MAP SIZE : ", new_h.get_shape()

            new_state = tf.nn.rnn_cell.LSTMStateTuple(f_u, new_h)
            return new_h, new_state

    # This is a P-MSTRNN layer used for Prop. Slow layer only (the highest prop. layer (Prop. Slow)
    # Note that Prop. Slow layer is handled as P-MSTRNN layer with a size of 1 x 1 feature map
    # This is for the one without a lateral input (from visual pathway)
    def feed_input_propTop_noLateral(self, inputs_motor, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            prev_f_u, prev_f_h = state

            print '-' * 90
            print "INPUT (MOTOR) : ", inputs_motor.get_shape()
            mt_filter = [1, 1]
            mt_strides = [1, 1, 1, 1]
            print "FILTER SIZE, STRIDES : ", mt_filter, mt_strides
            f_u2 = _conv_linear_pmstrnn([inputs_motor], mt_filter, mt_strides, self.num_features, bias=False,
                                       paddingType='VALID', scope='MOTOR')  # ??
            print "AFTER CONV : ", f_u2.get_shape()

            print '-' * 90
            print "INPUT (CONV.VALID.RECURRNT) : ", prev_f_h.get_shape()
            rc_filter = [1, 1]
            rc_strides = [1, 1, 1, 1]
            print "FILTER SIZE, STRIDES : ", rc_filter, rc_strides
            f_u3 = _conv_linear_pmstrnn([prev_f_h], rc_filter, rc_strides, self.num_features, bias=False,
                                        paddingType='SAME', scope='RC')  # ??
            print "AFTER CONV : ", f_u3.get_shape()

            print '-' * 90
            f_u = (1 - self._eta) * prev_f_u + self._eta * (f_u2 + f_u3)
            new_h = self._activation(f_u)
            print "OUTPUT_SHAPE : ", rc_filter
            print "MAP SIZE : ", new_h.get_shape()

            new_state = tf.nn.rnn_cell.LSTMStateTuple(f_u, new_h)
            return new_h, new_state


def _state_size_with_prefix(state_size, prefix=None):
  """Helper function that enables int or TensorShape shape specification.

  This function takes a size specification, which can be an integer or a
  TensorShape, and converts it into a list of integers. One may specify any
  additional dimensions that precede the final state size specification.

  Args:
    state_size: TensorShape or int that specifies the size of a tensor.
    prefix: optional additional list of dimensions to prepend.

  Returns:
    result_state_size: list of dimensions the resulting tensor size.
  """
  result_state_size = tensor_shape.as_shape(state_size).as_list()
  if prefix is not None:
    if not isinstance(prefix, list):
      raise TypeError("prefix of _state_size_with_prefix should be a list.")
    result_state_size = prefix + result_state_size
  return result_state_size

class RNNCell(object):
  """Abstract object representing an RNN cell.

  The definition of cell in this package differs from the definition used in the
  literature. In the literature, cell refers to an object with a single scalar
  output. The definition in this package refers to a horizontal array of such
  units.

  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  tuple of integers, then it results in a tuple of `len(state_size)` state
  matrices, each with a column size corresponding to values in `state_size`.

  This module provides a number of basic commonly used RNN cells, such as
  LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
  of operators that allow add dropouts, projections, or embeddings for inputs.
  Constructing multi-layer cells is supported by the class `MultiRNNCell`,
  or by calling the `rnn` ops several times. Every `RNNCell` must have the
  properties below and and implement `__call__` with the following signature.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size x self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size x s] for s in self.state_size`.
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:

      - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.

    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = self.state_size
    if nest.is_sequence(state_size):
      state_size_flat = nest.flatten(state_size)
      zeros_flat = [
          array_ops.zeros(
              array_ops.pack(_state_size_with_prefix(s, prefix=[batch_size])),
              dtype=dtype)
          for s in state_size_flat]
      for s, z in zip(state_size_flat, zeros_flat):
        z.set_shape(_state_size_with_prefix(s, prefix=[None]))
      zeros = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=zeros_flat)
    else:
      zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
      zeros = array_ops.zeros(array_ops.pack(zeros_size), dtype=dtype)
      zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

    return zeros

#=============================================================================
# MTRNN part
# This part is based on ~/tensorflow/tensorflow/python/ops/rnn_cell.py
#=============================================================================
class BasicCTRNNCell(RNNCell):
  def __init__(self, num_units, input_size=None,
               state_is_tuple=True, activation=tanh_mod, tau=1.0):
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._tau = tau
    self._eta = 1.0 / tau


  @property
  def state_size(self):
    return (tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with vs.variable_scope(scope or type(self).__name__):
      prev_u, prev_y = state
      # Calculating the internal states
      u = _linear([inputs, prev_y], self._num_units, bias=True)
      u = (1 - self._eta) * prev_u + self._eta * u
      # Calculating the activation value
      y = self._activation(u)
      new_state = tf.nn.rnn_cell.LSTMStateTuple(u, y)

      return y, new_state


  def feed_input(self, inputs, state, input_m1_conv, convMotor_strides, convMotor_filter, scope=None):
    with vs.variable_scope(scope or type(self).__name__):
      if self._state_is_tuple:
        prev_u, prev_y = state
      else:
        prev_u, prev_y = array_ops.split(1, 2, state)

      u = _linear([inputs, prev_y], self._num_units, bias=True)

      print "INPUT (CONV.VALID) : ", input_m1_conv.get_shape()
      print "FILTER SIZE, STRIDES : ", convMotor_filter, convMotor_strides
      f_u = _conv_linear_pmstrnn([input_m1_conv], convMotor_filter, convMotor_strides, self._num_units, bias=False,
                                 paddingType='VALID', scope='CONV_VALID')  # ??
      print "AFTER CONV : ", f_u.get_shape()
      u2 = tf.reshape(f_u, [-1, self._num_units])


      # Calculating the internal states
      u = (1 - self._eta) * prev_u + self._eta * (u + u2)
      # Calculating the activation value
      y = self._activation(u)

      if self._state_is_tuple:
        new_state = tf.nn.rnn_cell.LSTMStateTuple(u, y)
      else:
        new_state = array_ops.concat(1, [u, y])
      return y, new_state




