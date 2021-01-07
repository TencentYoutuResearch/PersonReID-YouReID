#conding=utf-8

import numpy as np
import tensorflow as tf

# borrow from
# https://github.com/tensorflow/models/blob/master/research/object_detection/utils/learning_schedules.py

def multi_stepping(global_step, boundaries, rates, warmup=False):
  """Manually stepped learning rate schedule.
  This function provides fine grained control over learning rates.  One must
  specify a sequence of learning rates as well as a set of integer steps
  at which the current learning rate must transition to the next.  For example,
  if boundaries = [5, 10] and rates = [.1, .01, .001], then the learning
  rate returned by this function is .1 for global_step=0,...,4, .01 for
  global_step=5...9, and .001 for global_step=10 and onward.
  Args:
    global_step: int64 (scalar) tensor representing global step.
    boundaries: a list of global steps at which to switch learning
      rates.  This list is assumed to consist of increasing positive integers.
    rates: a list of (float) learning rates corresponding to intervals between
      the boundaries.  The length of this list must be exactly
      len(boundaries) + 1.
    warmup: Whether to linearly interpolate learning rate for steps in
      [0, boundaries[0]].
  Returns:
    If executing eagerly:
      returns a no-arg callable that outputs the (scalar)
      float tensor learning rate given the current value of global_step.
    If in a graph:
      immediately returns a (scalar) float tensor representing learning rate.
  Raises:
    ValueError: if one of the following checks fails:
      1. boundaries is a strictly increasing list of positive integers
      2. len(rates) == len(boundaries) + 1
      3. boundaries[0] != 0
  """
  if any([b < 0 for b in boundaries]) or any(
      [not isinstance(b, int) for b in boundaries]):
    raise ValueError('boundaries must be a list of positive integers')
  if any([bnext <= b for bnext, b in zip(boundaries[1:], boundaries[:-1])]):
    raise ValueError('Entries in boundaries must be strictly increasing.')
  if any([not isinstance(r, float) for r in rates]):
    raise ValueError('Learning rates must be floats')
  if len(rates) != len(boundaries) + 1:
    raise ValueError('Number of provided learning rates must exceed '
                     'number of boundary points by exactly 1.')

  if boundaries and boundaries[0] == 0:
    raise ValueError('First step cannot be zero.')

  if warmup and boundaries:
    slope = (rates[1] - rates[0]) * 1.0 / boundaries[0]
    warmup_steps = list(range(boundaries[0]))
    warmup_rates = [rates[0] + slope * step for step in warmup_steps]
    boundaries = warmup_steps + boundaries
    rates = warmup_rates + rates[1:]
  else:
    boundaries = [0] + boundaries
  num_boundaries = len(boundaries)

  rate_index = tf.reduce_max(tf.where(
      tf.greater_equal(global_step, boundaries),
      list(range(num_boundaries)),
      [0] * num_boundaries))
  return tf.reduce_sum(rates * tf.one_hot(rate_index, depth=num_boundaries),
                       name='learning_rate')


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
  """Cosine decay schedule with warm up period.
  Cosine annealing learning rate as described in:
    Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
    ICLR 2017. https://arxiv.org/abs/1608.03983
  In this schedule, the learning rate grows linearly from warmup_learning_rate
  to learning_rate_base for warmup_steps, then transitions to a cosine decay
  schedule.
  Args:
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
  Returns:
    If executing eagerly:
      returns a no-arg callable that outputs the (scalar)
      float tensor learning rate given the current value of global_step.
    If in a graph:
      immediately returns a (scalar) float tensor representing learning rate.
  Raises:
    ValueError: if warmup_learning_rate is larger than learning_rate_base,
      or if warmup_steps is larger than total_steps.
  """
  if total_steps < warmup_steps:
    raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')

  learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
      np.pi *
      (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
       ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
  if hold_base_rate_steps > 0:
      learning_rate = tf.where(
          global_step > warmup_steps + hold_base_rate_steps,
          learning_rate, learning_rate_base)
  if warmup_steps > 0:
      if learning_rate_base < warmup_learning_rate:
          raise ValueError('learning_rate_base must be larger or equal to '
                           'warmup_learning_rate.')
      slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
      warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
      learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                               learning_rate)
  return tf.where(global_step > total_steps, 0.0, learning_rate,
                  name='learning_rate')
