import numpy as np
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, BatchNormalization, Flatten, Dense, \
    Activation, Add, concatenate, SeparableConv2D
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
import pickle
import logging
import math
import copy
import argparse
from sklearn.metrics import *
from sklearn.datasets import load_diabetes

from imgaug import augmenters as iaa
import imgaug as ia

from imgaug import parameters as iap
from imgaug import random as iarandom
from imgaug.augmenters import meta
from imgaug.augmenters import arithmetic
from imgaug.augmenters import flip
from imgaug.augmenters import pillike
from imgaug.augmenters import size as sizelib


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#x_train = x_train.astype("float32")
#x_test = x_test.astype("float32")
#x_train = x_train / 255.0
#x_test = x_test / 255.0

verbose = 1
batch_size = 2048
min_acc = 0.35

train_ind_half = list(range(0, 20000))
val_ind_half = list(range(20000, 25000))

train_ind_full = list(range(0, 40000))
val_ind_full = list(range(40000, 50000))

class Individual:

    class DenseModule:
        class DenseBlock:

            def build_block(self, x):
                if self.chromosome[0] <= self.dense_block_prob:
                    nodes = int(math.ceil(self.chromosome[1] / 10.0)) * 10
                    x = Dense(nodes, name="Dense_{}_{}".format(x.shape[1], nodes))(x)
                    if self.chromosome[2] <= self.dense_order_act_prob:
                        if self.chromosome[3] <= 1:  # always include dense act
                            dist = np.cumsum(self.dense_act_type_weights / np.sum(self.dense_act_type_weights))
                            activation = 'relu'
                            for k in range(0, len(self.act_types)):
                                if self.chromosome[4] <= dist[k]:
                                    activation = self.act_types[k]
                                    break
                            if activation == 'leaky_relu':
                                x = tf.keras.layers.LeakyReLU()(x)
                            else:
                                x = Activation(activation)(x)

                        if self.chromosome[5] <= self.dense_bn_prob:
                            x = BatchNormalization()(x)
                    else:
                        if self.chromosome[5] <= self.dense_bn_prob:
                            x = BatchNormalization()(x)

                        if self.chromosome[3] <= self.dense_act_prob:  # always include dense act
                            dist = np.cumsum(self.dense_act_type_weights / np.sum(self.dense_act_type_weights))
                            activation = 'relu'
                            for k in range(0, len(self.act_types)):
                                if self.chromosome[4] <= dist[k]:
                                    activation = self.act_types[k]
                                    break
                            if activation == 'leaky_relu':
                                x = tf.keras.layers.LeakyReLU()(x)
                            else:
                                x = Activation(activation)(x)
                    if self.chromosome[6] <= self.dense_drop_prob:
                        alpha = np.round(self.chromosome[7], 2)
                        x = Dropout(alpha)(x)
                return x

            def update_beliefs(self):

                self._check_bounds()

                self.dense_block_prob = self.dense_beliefs[0]
                self.dense_order_act_prob = self.dense_beliefs[1]
                self.dense_act_prob = self.dense_beliefs[2]
                self.dense_bn_prob = self.dense_beliefs[3]
                self.dense_drop_prob = self.dense_beliefs[4]
                self.dense_drop_max_alpha = self.dense_beliefs[5]
                self.dense_drop_min_alpha = self.dense_beliefs[6]

            def _check_bounds(self):
                for i in range(0, len(self.upper_bound)):
                    if self.chromosome[i] > self.upper_bound[i]:
                        self.chromosome[i] = self.upper_bound[i]
                    elif self.chromosome[i] < self.lower_bound[i]:
                        self.chromosome[i] = self.lower_bound[i]

                for i in range(0, len(self.dense_beliefs)):

                    if self.dense_beliefs[i] < 0.2:
                        self.dense_beliefs[i] = 0.2
                    elif self.dense_beliefs[i] > 0.8:
                        self.dense_beliefs[i] = 0.8

                    if i == len(self.dense_beliefs) - 2:
                        if self.dense_beliefs[i] < 0.5:
                            self.dense_beliefs[i] = 0.5
                    if i == len(self.dense_beliefs) - 1:
                        if self.dense_beliefs[i] > 0.49:
                            self.dense_beliefs[i] = 0.49

                for i in range(0, len(self.dense_act_type_weights)):
                    if self.dense_act_type_weights[i] < 0.2:
                        self.dense_act_type_weights[i] = 0.2
                    elif self.dense_act_type_weights[i] > 0.8:
                        self.dense_act_type_weights[i] = 0.8

            def __init__(self, min_node, max_node):
                self.max_node = max_node
                self.min_node = min_node
                self.act_types = ['relu', 'leaky_relu', 'selu', 'elu']
                self.dense_block_prob = np.random.uniform(0.2, 0.8)
                self.dense_order_act_prob = np.random.uniform(0.2, 0.8)
                self.dense_act_prob = np.random.uniform(0.2, 0.8)
                self.dense_act_type_weights = np.random.uniform(0.2, 0.8, 4)
                self.dense_bn_prob = np.random.uniform(0.2, 0.8)
                self.dense_drop_prob = np.random.uniform(0.2, 0.8)
                self.dense_drop_max_alpha = np.random.uniform(0.5, 0.8)
                self.dense_drop_min_alpha = np.random.uniform(0.2, 0.49)

                self.dense_beliefs = np.asarray([self.dense_block_prob, self.dense_order_act_prob, self.dense_act_prob,
                                      self.dense_bn_prob, self.dense_drop_prob,
                                      self.dense_drop_max_alpha, self.dense_drop_min_alpha])
                self.upper_bound = []
                self.lower_bound = []

                # include block
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # dense nodes
                self.upper_bound.append(self.max_node)
                self.lower_bound.append(self.min_node)

                # dense ordering of act and bn
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # dense act
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # act type
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # dense batchnorm
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # dense include drop
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # dense drop alpha
                self.upper_bound.append(self.dense_drop_max_alpha)
                self.lower_bound.append(self.dense_drop_min_alpha)

                self.chromosome = np.random.uniform(self.lower_bound, self.upper_bound)

        def __init__(self, min_blocks, max_blocks, min_node, max_node):
            self.max_node = max_node
            self.min_node = min_node
            self.min_blocks = min_blocks
            self.max_blocks = max_blocks
            self.dense_blocks = []
            for i in range(0, max_blocks):
                self.dense_blocks.append(self.DenseBlock(min_node=min_node, max_node=max_node))

        def build_module(self, x):

            for block in self.dense_blocks:
                x = block.build_block(x)
            return x

    class CnnModule:
        class CnnBlock:
            def _build_block_strides(self, x, filter, module_index, block_index):
                if self.chromosome[1] <= self.prob_inception:

                    if self.chromosome[3] <= self.prob_xception:  # xception

                        branch_0 = SeparableConv2D(int(filter / 4), (1, 1), strides=(2, 2), padding='same',
                                                   name="MODULE{}_STRIDES_Xception_branch0".format(module_index,
                                                                                                   block_index))(
                            x)
                        branch_1 = SeparableConv2D(int(filter / 4), (1, 1), padding='same',
                                                   name="MODULE{}_STRIDES_Xception_branch1_1x1".format(module_index,
                                                                                                       block_index))(
                            x)
                        branch_1 = SeparableConv2D(int(filter / 4), (3, 3), strides=(2, 2),
                                                   padding='same',
                                                   name="MODULE{}_STRIDES_Xception_branch1_3x3".format(module_index,
                                                                                                       block_index))(
                            branch_1)

                        branch_2 = SeparableConv2D(int(filter / 4), (1, 1), padding='same',
                                                   name="MODULE{}_STRIDES_Xception_branch2_1x1".format(module_index,
                                                                                                       block_index))(
                            x)
                        branch_2 = SeparableConv2D(int(filter / 4), (3, 3), padding='same',
                                                   name="MODULE{}_STRIDES_Xception_branch2_3x3_1".format(module_index,
                                                                                                         block_index))(
                            branch_2)
                        branch_2 = SeparableConv2D(int(filter / 4), (3, 3), strides=(2, 2),
                                                   padding='same',
                                                   name="MODULE{}_STRIDES_Xception_branch2_3x3_2".format(module_index,
                                                                                                         block_index))(
                            branch_2)

                        if self.chromosome[5] <= self.prob_xception_max:
                            branch_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                                                    name="MODULE{}_STRIDES_Xception_branch3_AvgPool".format(
                                                        module_index,
                                                        block_index))(
                                x)
                        else:
                            branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                                        name="MODULE{}_STRIDES_Xception_branch3_AvgPool".format(
                                                            module_index,
                                                            block_index))(
                                x)
                        branch_3 = SeparableConv2D(int(filter / 4), (1, 1), strides=(2, 2),
                                                   padding='same',
                                                   name="MODULE{}_STRIDES_Xception_branch3_1x1".format(module_index,
                                                                                                       block_index))(
                            branch_3)
                    else:
                        branch_0 = Conv2D(int(filter / 4), (1, 1), strides=(2, 2), padding='same',
                                          name="MODULE{}_STRIDES_Inception_branch0".format(module_index, block_index))(
                            x)
                        branch_1 = Conv2D(int(filter / 4), (1, 1), padding='same',
                                          name="MODULE{}_STRIDES_Inception_branch1_1x1".format(module_index,
                                                                                               block_index))(
                            x)
                        branch_1 = Conv2D(int(filter / 4), (3, 3), strides=(2, 2), padding='same',
                                          name="MODULE{}_STRIDES_Inception_branch1_3x3".format(module_index,
                                                                                               block_index))(
                            branch_1)

                        branch_2 = Conv2D(int(filter / 4), (1, 1), padding='same',
                                          name="MODULE{}_STRIDES_Inception_branch2_1x1".format(module_index,
                                                                                               block_index))(
                            x)
                        branch_2 = Conv2D(int(filter / 4), (3, 3), padding='same',
                                          name="MODULE{}_STRIDES_Inception_branch2_3x3_1".format(module_index,
                                                                                                 block_index))(
                            branch_2)
                        branch_2 = Conv2D(int(filter / 4), (3, 3), strides=(2, 2), padding='same',
                                          name="MODULE{}_STRIDES_Inception_branch2_3x3_2".format(module_index,
                                                                                                 block_index))(
                            branch_2)

                        if self.chromosome[4] <= self.prob_inception_max:
                            branch_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                                                    name="MODULE{}_STRIDES_Inception_branch3_AvgPool".format(
                                                        module_index,
                                                        block_index))(
                                x)
                        else:
                            branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                                        name="MODULE{}_STRIDES_Inception_branch3_AvgPool".format(
                                                            module_index,
                                                            block_index))(
                                x)
                        branch_3 = Conv2D(int(filter / 4), (1, 1), strides=(2, 2), padding='same',
                                          name="MODULE{}_STRIDES_Inception_branch3_1x1".format(module_index,
                                                                                               block_index))(
                            branch_3)

                    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
                else:
                    if self.chromosome[2] <= self.prob_bottle:
                        x = Conv2D(int(filter / 4), (1, 1), padding='same', strides=(2, 2),
                                   name="STRIDES_bottle1_MODULE{}_BLOCK{}".format(module_index, block_index))(x)
                        x = Conv2D(int(filter / 4), (3, 3), padding='same',
                                   name="STRIDES_bottle2_MODULE{}_BLOCK{}".format(module_index, block_index))(x)
                        x = Conv2D(filter, (1, 1), padding='same',
                                   name="STRIDES_bottle3_MODULE{}_BLOCK{}".format(module_index, block_index))(x)
                    else:
                        x = Conv2D(filter, (3, 3), padding='same', strides=(2, 2),
                                   name="MODULE{}_STRIDES".format(module_index, block_index))(x)
                return x

            def _build_block_shortcuts(self, x, shortcuts, prev_layer, block_index, module_index):
                if self.chromosome[0] <= self.block_prob:
                    if len(shortcuts) != 0:
                        shorts = []
                        for key, value in shortcuts.items():
                            if key == prev_layer:
                                continue
                            shorts.append(value)
                        if len(shorts) != 0:
                            shorts.append(x)
                            x = Add()(shorts)

                            if self.chromosome[13] <= self.short_order_prob:
                                if self.chromosome[14] <= self.short_act_prob:

                                    dist = np.cumsum(
                                        self.short_act_type_weights / np.sum(self.short_act_type_weights))
                                    activation = 'relu'
                                    for k in range(0, len(self.act_types)):
                                        if self.chromosome[15] <= dist[k]:
                                            activation = self.act_types[k]
                                            break

                                    if activation == 'leaky_relu':
                                        x = tf.keras.layers.LeakyReLU(
                                            name="Short_M{}_B{}_Act_{}".format(module_index, block_index, activation))(
                                            x)
                                    else:
                                        x = Activation(activation,
                                                       name="Short_M{}_B{}_Act_{}".format(module_index, block_index,
                                                                                          activation))(
                                            x)

                                if self.chromosome[16] <= self.short_bn_prob:
                                    x = BatchNormalization(
                                        name="SHORT_M{}_B{}_Batch".format(module_index, block_index))(x)

                            else:
                                if self.chromosome[16] <= self.short_bn_prob:
                                    x = BatchNormalization(
                                        name="SHORT_M{}_B{}_Batch".format(module_index, block_index))(x)

                                if self.chromosome[14] <= self.short_act_prob:

                                    dist = np.cumsum(
                                        self.short_act_type_weights / np.sum(self.short_act_type_weights))
                                    activation = 'relu'
                                    for k in range(0, len(self.act_types)):
                                        if self.chromosome[15] <= dist[k]:
                                            activation = self.act_types[k]
                                            break

                                    if activation == 'leaky_relu':
                                        x = tf.keras.layers.LeakyReLU(
                                            name="Short_M{}_B{}_Act_{}".format(module_index, block_index, activation))(
                                            x)
                                    else:
                                        x = Activation(activation,
                                                       name="Short_M{}_B{}_Act_{}".format(module_index, block_index,
                                                                                          activation))(
                                            x)

                            if self.chromosome[17] <= self.short_drop_prob:
                                alpha = np.round(self.chromosome[18], 2)
                                x = Dropout(alpha,
                                            name="SHORT_M{}_B{}_Drop{}".format(module_index, block_index, alpha))(x)
                return x

            def _build_block_middle(self, x, filter, shortcuts, block_index, module_index):
                if self.chromosome[1] <= self.prob_inception:

                    if self.chromosome[3] <= self.prob_xception:  # xception

                        branch_0 = SeparableConv2D(int(filter / 4), (1, 1), padding='same',
                                                   name="MODULE{}_BLOCK{}_Xception_branch0".format(module_index,
                                                                                                   block_index))(
                            x)
                        branch_1 = SeparableConv2D(int(filter / 4), (1, 1), padding='same',
                                                   name="MODULE{}_BLOCK{}_Xception_branch1_1x1".format(module_index,
                                                                                                       block_index))(
                            x)
                        branch_1 = SeparableConv2D(int(filter / 4), (3, 3), padding='same',
                                                   name="MODULE{}_BLOCK{}_Xception_branch1_3x3".format(module_index,
                                                                                                       block_index))(
                            branch_1)

                        branch_2 = SeparableConv2D(int(filter / 4), (1, 1), padding='same',
                                                   name="MODULE{}_BLOCK{}_Xception_branch2_1x1".format(module_index,
                                                                                                       block_index))(
                            x)
                        branch_2 = SeparableConv2D(int(filter / 4), (3, 3), padding='same',
                                                   name="MODULE{}_BLOCK{}_Xception_branch2_3x3_1".format(module_index,
                                                                                                         block_index))(
                            branch_2)
                        branch_2 = SeparableConv2D(int(filter / 4), (3, 3), padding='same',
                                                   name="MODULE{}_BLOCK{}_Xception_branch2_3x3_2".format(module_index,
                                                                                                         block_index))(
                            branch_2)

                        if self.chromosome[5] <= self.prob_xception_max:
                            branch_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                                                    name="MODULE{}_BLOCK{}_Xception_branch3_AvgPool".format(
                                                        module_index,
                                                        block_index))(
                                x)
                        else:
                            branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                                        name="MODULE{}_BLOCK{}_Xception_branch3_AvgPool".format(
                                                            module_index,
                                                            block_index))(
                                x)
                        branch_3 = SeparableConv2D(int(filter / 4), (1, 1), padding='same',
                                                   name="MODULE{}_BLOCK{}_Xception_branch3_1x1".format(module_index,
                                                                                                       block_index))(
                            branch_3)
                    else:
                        branch_0 = Conv2D(int(filter / 4), (1, 1), padding='same',
                                          name="MODULE{}_BLOCK{}_Inception_branch0".format(module_index, block_index))(
                            x)
                        branch_1 = Conv2D(int(filter / 4), (1, 1), padding='same',
                                          name="MODULE{}_BLOCK{}_Inception_branch1_1x1".format(module_index,
                                                                                               block_index))(
                            x)
                        branch_1 = Conv2D(int(filter / 4), (3, 3), padding='same',
                                          name="MODULE{}_BLOCK{}_Inception_branch1_3x3".format(module_index,
                                                                                               block_index))(
                            branch_1)

                        branch_2 = Conv2D(int(filter / 4), (1, 1), padding='same',
                                          name="MODULE{}_BLOCK{}_Inception_branch2_1x1".format(module_index,
                                                                                               block_index))(
                            x)
                        branch_2 = Conv2D(int(filter / 4), (3, 3), padding='same',
                                          name="MODULE{}_BLOCK{}_Inception_branch2_3x3_1".format(module_index,
                                                                                                 block_index))(
                            branch_2)
                        branch_2 = Conv2D(int(filter / 4), (3, 3), padding='same',
                                          name="MODULE{}_BLOCK{}_Inception_branch2_3x3_2".format(module_index,
                                                                                                 block_index))(
                            branch_2)

                        if self.chromosome[4] <= self.prob_inception_max:
                            branch_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                                                    name="MODULE{}_BLOCK{}_Inception_branch3_AvgPool".format(
                                                        module_index,
                                                        block_index))(
                                x)
                        else:
                            branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same',
                                                        name="MODULE{}_BLOCK{}_Inception_branch3_AvgPool".format(
                                                            module_index,
                                                            block_index))(
                                x)
                        branch_3 = Conv2D(int(filter / 4), (1, 1), padding='same',
                                          name="MODULE{}_BLOCK{}_Inception_branch3_1x1".format(module_index,
                                                                                               block_index))(
                            branch_3)

                    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=-1)
                else:
                    if self.chromosome[2] <= self.prob_bottle:
                        x = Conv2D(int(filter / 4), (1, 1), padding='same',
                                   name="bottle1_MODULE{}_BLOCK{}".format(module_index, block_index))(x)
                        x = Conv2D(int(filter / 4), (3, 3), padding='same',
                                   name="bottle2_MODULE{}_BLOCK{}".format(module_index, block_index))(x)
                        x = Conv2D(filter, (1, 1), padding='same',
                                   name="bottle3_MODULE{}_BLOCK{}".format(module_index, block_index))(x)
                    else:
                        x = Conv2D(filter, (3, 3), padding='same',
                                   name="MODULE{}_BLOCK{}".format(module_index, block_index))(x)

                if self.chromosome[6] <= self.short_prob:
                    shortcuts[block_index] = x

                if self.chromosome[7] <= self.order_prob:
                    if self.chromosome[8] <= self.act_prob:

                        dist = np.cumsum(
                            self.short_act_type_weights / np.sum(self.short_act_type_weights))
                        activation = 'relu'
                        for k in range(0, len(self.act_types)):
                            if self.chromosome[9] <= dist[k]:
                                activation = self.act_types[k]
                                break

                        if activation == 'leaky_relu':
                            x = tf.keras.layers.LeakyReLU(
                                name="M{}_B{}_Act_{}".format(module_index, block_index, activation))(x)
                        else:
                            x = Activation(activation,
                                           name="M{}_B{}_Act_{}".format(module_index, block_index, activation))(
                                x)

                    if self.chromosome[10] <= self.bn_prob:
                        x = BatchNormalization(name="M{}_B{}_Batch".format(module_index, block_index))(x)

                else:
                    if self.chromosome[10] <= self.bn_prob:
                        x = BatchNormalization(name="M{}_B{}_Batch".format(module_index, block_index))(x)

                    if self.chromosome[8] <= self.act_prob:

                        dist = np.cumsum(
                            self.short_act_type_weights / np.sum(self.short_act_type_weights))
                        activation = 'relu'
                        for k in range(0, len(self.act_types)):
                            if self.chromosome[9] <= dist[k]:
                                activation = self.act_types[k]
                                break

                        if activation == 'leaky_relu':
                            x = tf.keras.layers.LeakyReLU(
                                name="M{}_B{}_Act_{}".format(module_index, block_index, activation))(x)
                        else:
                            x = Activation(activation,
                                           name="M{}_B{}_Act_{}".format(module_index, block_index,
                                                                        activation))(
                                x)

                if self.chromosome[11] <= self.drop_prob:
                    alpha = np.round(self.chromosome[12], 2)
                    x = Dropout(alpha, name="M{}_B{}_Drop{}".format(module_index, block_index, alpha))(x)
                return x

            def build_block(self, x, filter, shortcuts, prev_layer, block_index, module_index,
                            build_strides=False):

                # not the first block in module, so get shortcuts
                if block_index != 0:
                    x = self._build_block_shortcuts(shortcuts=shortcuts, prev_layer=prev_layer, block_index=block_index,
                                                   module_index=module_index, x=x)
                if build_strides:  # last block in module, so use strides
                    x = self._build_block_strides(x=x, filter=filter, module_index=module_index, block_index=block_index)
                else:  # middle block
                    x = self._build_block_middle(x=x, filter=filter, shortcuts=shortcuts, block_index=block_index,
                                                module_index=module_index)

                return x

            def update_beliefs(self):

                self._check_bounds()

                self.block_prob = self.beliefs_probs[0]
                self.short_prob = self.beliefs_probs[1]
                self.prob_inception = self.beliefs_probs[2]
                self.prob_xception = self.beliefs_probs[3]
                self.prob_bottle = self.beliefs_probs[4]
                self.prob_inception_max = self.beliefs_probs[5]
                self.prob_xception_max = self.beliefs_probs[6]
                self.order_prob = self.beliefs_probs[7]
                self.act_prob = self.beliefs_probs[8]
                self.act_type_weights = self.beliefs_act_weights[0]
                self.bn_prob = self.beliefs_probs[9]
                self.drop_prob = self.beliefs_probs[10]
                self.max_alphas = self.beliefs_probs[11]
                self.min_alphas = self.beliefs_probs[12]
                self.short_order_prob = self.beliefs_probs[13]
                self.short_act_prob = self.beliefs_probs[14]
                self.short_act_type_weights = self.beliefs_act_weights[1]
                self.short_bn_prob = self.beliefs_probs[15]
                self.short_drop_prob = self.beliefs_probs[16]
                self.short_max_alphas = self.beliefs_probs[17]
                self.short_min_alphas = self.beliefs_probs[18]

            def _check_bounds(self):
                for i in range(0, len(self.upper_bound)):
                    if self.chromosome[i] > self.upper_bound[i]:
                        self.chromosome[i] = self.upper_bound[i]
                    elif self.chromosome[i] < self.lower_bound[i]:
                        self.chromosome[i] = self.lower_bound[i]

                for i in range(0, len(self.beliefs_probs)):

                    if self.beliefs_probs[i] < 0.2:
                        self.beliefs_probs[i] = 0.2
                    elif self.beliefs_probs[i] > 0.8:
                        self.beliefs_probs[i] = 0.8

                    if i == 11 or i == 17:
                        if self.beliefs_probs[i] < 0.5:
                            self.beliefs_probs[i] = 0.5
                    if i == 12 or i == 18:
                        if self.beliefs_probs[i] > 0.49:
                            self.beliefs_probs[i] = 0.49

                for i in range(0, len(self.beliefs_act_weights)):
                    for j in range(0, len(self.beliefs_act_weights[i])):
                        if self.beliefs_act_weights[i][j] < 0.2:
                            self.beliefs_act_weights[i][j] = 0.2
                        elif self.beliefs_act_weights[i][j] > 0.8:
                            self.beliefs_act_weights[i][j] = 0.8

            def __init__(self):
                self.block_prob = np.random.uniform(0.2, 0.8)
                self.short_prob = np.random.uniform(0.2, 0.8)
                self.prob_inception = np.random.uniform(0.2, 0.8)
                self.prob_xception = np.random.uniform(0.2, 0.8)
                self.prob_bottle = np.random.uniform(0.2, 0.8)
                self.prob_inception_max = np.random.uniform(0.2, 0.8)
                self.prob_xception_max = np.random.uniform(0.2, 0.8)
                self.order_prob = np.random.uniform(0.2, 0.8)
                self.act_prob = np.random.uniform(0.2, 0.8)
                self.act_type_weights = np.random.uniform(0.2, 0.8, 4)
                self.act_types = ['relu', 'leaky_relu', 'selu', 'elu']
                self.bn_prob = np.random.uniform(0.2, 0.8)
                self.drop_prob = np.random.uniform(0.2, 0.8)
                self.max_alphas = np.random.uniform(0.5, 0.8)
                self.min_alphas = np.random.uniform(0.2, 0.49)
                self.short_order_prob = np.random.uniform(0.2, 0.8)
                self.short_act_prob = np.random.uniform(0.2, 0.8)
                self.short_act_type_weights = np.random.uniform(0.2, 0.8, 4)
                self.short_bn_prob = np.random.uniform(0.2, 0.8)
                self.short_drop_prob = np.random.uniform(0.2, 0.8)
                self.short_max_alphas = np.random.uniform(0.5, 0.8)
                self.short_min_alphas = np.random.uniform(0.2, 0.49)

                self.beliefs_probs = np.asarray([
                    self.block_prob, self.short_prob, self.prob_inception, self.prob_xception, self.prob_bottle,
                    self.prob_inception_max, self.prob_xception_max, self.order_prob, self.act_prob, self.bn_prob,
                    self.drop_prob, self.max_alphas, self.min_alphas, self.short_order_prob, self.short_act_prob,
                    self.short_bn_prob, self.short_drop_prob, self.short_max_alphas, self.short_min_alphas])

                self.beliefs_act_weights = np.asarray([self.act_type_weights, self.short_act_type_weights])


                self.upper_bound = []
                self.lower_bound = []
                # include block
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # make conv2d or inception?
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # if conv2d - include bottleneck?
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # if inception - make inception or xception?
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # if inception - max or avg pool?
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # if xception - max or avg pool?
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # skip connection
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # ordering of bn and act:
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # include act
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # act type
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # include bn
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # include drop
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # drop alpha
                self.upper_bound.append(self.max_alphas)
                self.lower_bound.append(self.min_alphas)

                # short ordering of bn and act:
                self.upper_bound.append(1)
                self.lower_bound.append(0)
                # shortcut act
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # act type
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # shortcut batchnorm
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # short include drop
                self.upper_bound.append(1)
                self.lower_bound.append(0)

                # short drop alpha
                self.upper_bound.append(self.short_max_alphas)
                self.lower_bound.append(self.short_min_alphas)

                self.chromosome = np.random.uniform(self.lower_bound, self.upper_bound)

        def __init__(self, min_blocks, max_blocks, filter, module_index):
            self.min_blocks = min_blocks
            self.module_index = module_index
            self.filter = filter
            self.max_blocks = max_blocks
            self.prob_include = np.random.uniform(0.2, 0.8)
            self.cnn_blocks = []
            for i in range(0, max_blocks):
                self.cnn_blocks.append(self.CnnBlock())

        def build_module(self, x):
            shortcuts = {}
            block_index = 0
            prev_layer = 0
            for i in range(0, self.max_blocks):
                block = self.cnn_blocks[i]
                if i < (self.min_blocks-1) or block.chromosome[0] <= block.block_prob or i == (self.max_blocks-1):
                    if i == (self.max_blocks-1):
                        build_strides = True
                    else:
                        build_strides = False

                    x = block.build_block(x, filter=self.filter, prev_layer=prev_layer, shortcuts=shortcuts,
                                          module_index=self.module_index, block_index=block_index,
                                          build_strides=build_strides)
                    block_index += 1
                    prev_layer = i

            return x

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255.0)(inputs)
        free_module_index = 0
        for i in range(0, self.num_modules):
            module = self.cnn_modules[i]
            if i < self.min_modules:
                x = module.build_module(x)
            else:
                if module.prob_include < self.free_modules_prob[free_module_index]:
                    x = module.build_module(x)
                free_module_index += 1

        x = Flatten()(x)
        x = self.dense_module.build_module(x)
        outputs = Dense(self.num_output, activation=self.output_act,
                        name="Dense_{}_{}".format(x.shape[1], self.num_output))(x)
        model = tf.keras.Model(inputs, outputs)
        self.num_param = model.count_params()
        return model

    def __init__(self, cnn_min_blocks, cnn_max_blocks, dense_min_blocks, dense_max_blocks,
                 min_modules, max_node, min_node, input_shape, filters, num_output, output_act):
        self.num_modules = len(filters)
        self.dense_max_blocks = dense_max_blocks
        self.dense_min_blocks = dense_min_blocks
        self.cnn_max_blocks = cnn_max_blocks
        self.cnn_min_blocks = cnn_min_blocks
        self.num_output = num_output
        self.output_act = output_act
        self.min_modules = min_modules
        self.max_node = max_node
        self.min_node = min_node
        self.input_shape = input_shape
        self.filters = filters
        self.cnn_modules = []
        self.num_free_modules = self.num_modules - self.min_modules
        self.free_modules_prob = np.random.uniform(0, 1, self.num_free_modules)
        self.age = 0
        self.gen = 0
        self.num_param = 0
        for i in range(0, self.num_modules):
            self.cnn_modules.append(self.CnnModule(min_blocks=cnn_min_blocks[i], max_blocks=cnn_max_blocks[i], module_index=i,
                                               filter=filters[i]))
        self.dense_module = self.DenseModule(min_blocks=dense_min_blocks, max_blocks=dense_max_blocks, min_node=min_node,
                                             max_node=max_node)

class SensitivityAnalysis:

    def  __init__(self, fitness_function_half, fitness_function_full, cnn_min_blocks, cnn_max_blocks, dense_min_blocks, dense_max_blocks,
                 min_modules, max_node, min_node, input_shape, filters, num_output, output_act, init_pop_size,
                  save_dir):
        self.num_modules = len(filters)
        self.dense_max_blocks = dense_max_blocks
        self.dense_min_blocks = dense_min_blocks
        self.cnn_max_blocks = cnn_max_blocks
        self.cnn_min_blocks = cnn_min_blocks
        self.num_output = num_output
        self.output_act = output_act
        self.min_modules = min_modules
        self.max_node = max_node
        self.min_node = min_node
        self.input_shape = input_shape
        self.filters = filters
        self.fitness_function_half = fitness_function_half
        self.fitness_function_full = fitness_function_full
        self.init_pop_size = init_pop_size
        self.init_individuals = []
        self.init_fitness = []
        self.save_dir = save_dir

    def fit(self):
        msg = "STARTING INITIAL POPULATION"
        print(msg)
        logging.info(msg)
        index = 0
        while index < self.init_pop_size:

            individual = Individual(cnn_min_blocks=self.cnn_min_blocks, cnn_max_blocks=self.cnn_max_blocks,
                                    dense_min_blocks=self.dense_min_blocks, dense_max_blocks=self.dense_max_blocks,
                 min_modules=self.min_modules, max_node=self.max_node, min_node=self.min_node, input_shape=self.input_shape,
                                    filters=self.filters, num_output=self.num_output, output_act=self.output_act)

            model = individual.build_model()
            result1 = self.fitness_function_half(model)

            if result1 is None:
                msg = " MODEL ARCHITECTURE FAILED..."
                print(msg)
                logging.info(msg)
                continue

            num_param = model.count_params()
            msg = " MODEL {} - Num Param: {}".format(index, num_param)
            print(msg)
            logging.info(msg)

            t_a, t_l, v_a, v_l = result1
            msg = "   Half: " \
                  "     t_a: {}\n" \
                  "     t_l: {}\n" \
                  "     v_a: {}\n" \
                  "     v_l: {}\n".format(t_a, t_l, v_a, v_l)

            print(msg)
            logging.info(msg)

            model = individual.build_model()
            result2 = self.fitness_function_full(model)
            t_a, t_l, v_a, v_l = result2
            msg = "   Full: " \
                  "     t_a: {}\n" \
                  "     t_l: {}\n" \
                  "     v_a: {}\n" \
                  "     v_l: {}\n".format(t_a, t_l, v_a, v_l)
            print(msg)
            logging.info(msg)
            self.init_individuals.append(individual)
            self.init_fitness.append([result1, result2])
            index += 1

        self.init_fitness = np.asarray(self.init_fitness)
        self.init_individuals = np.asarray(self.init_individuals)

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

class DegenerateModelDetection(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= 4:
            if logs['accuracy'] <= min_acc:
                self.model.stop_training = True

def fitness_function_half(model, epochs=150):
    opt = tf.keras.optimizers.Adam(clipnorm=1.0)
    timeCallback = TimeHistory()
    term = TerminateOnNaN()
    deg = DegenerateModelDetection()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    callback = [EarlyStopping(monitor='loss', patience=25, restore_best_weights=True),
                EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True),
                EarlyStopping(monitor='val_loss', patience=75, restore_best_weights=True),
                EarlyStopping(monitor='accuracy', patience=15, restore_best_weights=True),
                timeCallback,  deg,
                term]

    history = model.fit(x_train[train_ind_half], y_train[train_ind_half], batch_size=batch_size, epochs=epochs,
            verbose=verbose, callbacks=callback, validation_data=(x_train[val_ind_half], y_train[val_ind_half]))


    if np.nanmax(history.history['accuracy']) <= min_acc:
        return None

    v_a = []
    v_l = []
    t_a = []
    t_l = []
    for i in [0, 2, 4, 9, 24, 49]:
        v_a.append(history.history['val_accuracy'][i])
        v_l.append(history.history['val_loss'][i])
        t_a.append(history.history['accuracy'][i])
        t_l.append(history.history['loss'][i])
    v_a.append(np.nanmax(history.history['val_accuracy']))
    v_l.append(np.nanmin(history.history['val_loss']))
    t_a.append(np.nanmax(history.history['accuracy']))
    t_l.append(np.nanmin(history.history['loss']))
    return t_a, t_l, v_a, v_l

def fitness_function_full(model, epochs=150):
    opt = tf.keras.optimizers.Adam(clipnorm=1.0)
    timeCallback = TimeHistory()
    term = TerminateOnNaN()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    callback = [EarlyStopping(monitor='loss', patience=25, restore_best_weights=True),
                EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True),
                EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True),
                EarlyStopping(monitor='accuracy', patience=15, restore_best_weights=True),
                timeCallback,
                term]

    history = model.fit(x_train[train_ind_full], y_train[train_ind_full], batch_size=batch_size, epochs=epochs,
                        verbose=verbose, callbacks=callback,
                        validation_data=(x_train[val_ind_full], y_train[val_ind_full]))

    v_a = []
    v_l = []
    t_a = []
    t_l = []
    for i in [0, 2, 4, 9, 24, 49]:
        v_a.append(history.history['val_accuracy'][i])
        v_l.append(history.history['val_loss'][i])
        t_a.append(history.history['accuracy'][i])
        t_l.append(history.history['loss'][i])
    v_a.append(np.nanmax(history.history['val_accuracy']))
    v_l.append(np.nanmin(history.history['val_loss']))
    t_a.append(np.nanmax(history.history['accuracy']))
    t_l.append(np.nanmin(history.history['loss']))
    return t_a, t_l, v_a, v_l

# half - train loss, train acc, val loss, val acc at iter 1, 3, 5, 10, 25, 50, convergence
def create_parser():
    '''
    Create a command line parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='Differential Evolution')

    parser.add_argument('--cnn_min_blocks', type=int, default=2, help='Minimum # of blocks per CNN Module')
    parser.add_argument('--cnn_max_blocks', type=int, default=4, help='Maximum # of blocks per CNN Module')
    parser.add_argument('--dense_max_blocks', type=int, default=2, help='Maximum # of blocks per Dense Module')
    parser.add_argument('--dense_min_blocks', type=int, default=0, help='Minimum # of blocks per Dense Module')
    parser.add_argument('--min_nodes', type=int, default=100, help='Min Number of hidden units')
    parser.add_argument('--max_nodes', type=int, default=1000, help='Max Number of hidden units')
    parser.add_argument('--init_pop_size', type=int, default=15, help='Initial Population Size')
    parser.add_argument('--num_output', type=int, default=10, help='Number of output units')
    parser.add_argument('--output_act', type=str, default="softmax", help='Number of hidden units')
    parser.add_argument('--logs_file', type=str, default='sensitivity_logs.log', help='Output File For Logging')
    parser.add_argument('--save_dir', type=str, default='sensitivity_analysis', help='Save Directory for saving Model weights')
    parser.add_argument('--algo_save_file', type=str, default='sensitivity_analysis',
                        help='Save File for Algorithm')
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    filters = [64, 128, 256, 512]
    num_modules = len(filters)
    cnn_min_blocks = [args.cnn_min_blocks] * num_modules
    cnn_max_blocks = [args.cnn_max_blocks] * num_modules
    dense_max_blocks = args.dense_max_blocks
    dense_min_blocks = args.dense_min_blocks
    input_shape = (32, 32, 3)
    max_nodes = args.max_nodes
    min_nodes = args.min_nodes
    num_output = args.num_output
    output_act = args.output_act
    init_pop_size = args.init_pop_size
    min_module = num_modules - 1
    save_dir = args.save_dir
    logs_file = args.logs_file
    algo_save_file = args.algo_save_file
    logging.basicConfig(filename=logs_file, level=logging.DEBUG)

    start = time.time()
    msg = "--- Starting Sensitivity Analysis ---"
    logging.info(msg)
    print(msg)

    sa = SensitivityAnalysis(fitness_function_half=fitness_function_half,
                                    fitness_function_full=fitness_function_full, dense_max_blocks=dense_max_blocks,
                                 cnn_min_blocks=cnn_min_blocks, cnn_max_blocks=cnn_max_blocks, min_modules=min_module,
                                 max_node=max_nodes, min_node=min_nodes, input_shape=input_shape, num_output=num_output,
                                 output_act=output_act, filters=filters, init_pop_size=init_pop_size,
                                 dense_min_blocks=dense_min_blocks, save_dir=save_dir)


    sa.fit()

    pickle.dump(sa, open(save_dir+"/"+algo_save_file, "wb"))


    msg = "--- ENDING Sensitivity Analysis ---"
    print(msg)
    logging.info(msg)
    finish = time.time()
    msg = "--- Time Taken: {} min ---".format((finish-start)/60.0)
    logging.info(msg)
    print(msg)
