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

class GenericUnconstrainedProblem:
    """Generic Genetic Algorithm for Unconstrained Optimization Problems
    GenericUnconstrainedProblem is a simpler Genetic Algorithm than
    HyperParamUnconstrainedProblem by having fewer hyper-parameters, namely
    generation size, maximum iterations, and algorithm.
    GenericUnconstrainedProblem implements an "evolve" and "plot" method.
    Parameters
    -----------
    fitness_function : function pointer
        A pointer to a function that will evaluate and return the fitness of
        each individual in a population given their parameter values. The function
        should expect two parameters ``generation``, which will be a list of lists,
        where each sub list is an individual; and ``init_pop_print`` which is a
        boolean value defaulted to False which allows the user to print out
        statements during the initial population selection, if chosen (see
        examples for more info on this parameter). Lastly, the function should
        return a numpy array of the fitness values.
    upper_bound : list or numpy 1d array
        A list or numpy 1d array representing the upper bound of the domain for the
        unconstrained problem, where the first index of the list represents the upper
        bound for the first variable. For example, if x1=4, x2=4, x3=8 are the upper
        limits of the variables, then pass in ``[4, 4, 8]`` as the upper bound.
    lower_bound : list or numpy 1d array
        A list or numpy 1d array representing the lower bound of the domain for the
        unconstrained problem, where the first index of the list represents the lower
        bound for the first variable. For example, if x1=0, x2=-4, x3=1 are the lower
        limits of the variables, then pass in ``[0, -4, 1]`` as the lower bound.
    gen_size : int
        The number of individuals within each generation to perform evolution with.
    Attributes
    -----------
    gen : numpy 2D array
        A numpy 2D array of the individuals from the last generation of evolution.
    best_individual : numpy 1D array
        A numpy 1D array of the best individual from the last generation of evolution.
    best_fit : list
        A list of the best fitness values per generation of evolution.
    mean_fit : list
        A list of the mean fitness values per generation of evolution.
    best_values : list
        A list of the best individual per generation of evolution.
    """

    def __init__(self, fitness_function, upper_bound, lower_bound, gen_size):
        self.gen_size = gen_size
        self.fitness_function = fitness_function
        self.best_fit = []
        self.mean_fit = []
        self.best_values = []
        self.num_variables = len(upper_bound)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.total_bound = np.asarray(upper_bound) - np.asarray(lower_bound)
        self.domain = [upper_bound, lower_bound]
        self.gen = None
        self.sigma = None
        self.best_individual = None

    # mutates the current generation to create offspring using
    # the strategy parameters sigma
    def __mutation_self_adaptive(self, parent, strategy):
        tau = 1 / (np.sqrt(2 * np.sqrt(self.num_variables)))
        tau_prime = 1 / (np.sqrt(2 * self.num_variables))
        r = np.random.normal(0, 1, len(parent))
        child_sigma = strategy * np.exp(tau * r + tau_prime * r)
        # r = np.random.laplace(0, 1, len(parent))
        r = np.random.normal(0, 1, len(parent))
        # r = np.random.standard_cauchy(len(parent))
        child_value = np.copy(parent) + child_sigma * r
        return child_value, child_sigma

    def __crossover_method_1(self, par):
        return np.mean(par, axis=0)

    def __crossover_method_2(self, par):
        child = np.copy(par[0])
        n = np.shape(par[0])[0]
        random_nums = np.random.randint(low=0, high=len(par), size=n)
        for j in range(0, n):
            child[j] = par[random_nums[j]][j]
        return child

    def __reproduction_self_adapt(self, par, sigma, f_par, find_max):
        c1_values, c1_sigma = self.__mutation_self_adaptive(par, sigma)
        c2_values, c2_sigma = self.__mutation_self_adaptive(par, sigma)
        c3_values, c3_sigma = self.__mutation_self_adaptive(par, sigma)
        c4_values, c4_sigma = self.__mutation_self_adaptive(par, sigma)

        total_val = np.asarray([c1_values, c2_values, c3_values, c4_values])
        total_sigma = np.asarray([c1_sigma, c2_sigma, c3_sigma, c4_sigma])
        for i in range(0, 4):
            for j in range(0, len(c1_values)):
                if total_val[i][j] > self.upper_bound[j]:
                    total_val[i][j] = self.upper_bound[j]
                    total_sigma[i][j] *= 0.9
                elif total_val[i][j] < self.lower_bound[j]:
                    total_val[i][j] = self.lower_bound[j]
                    total_sigma[i][j] *= 0.9

        f = self.fitness_function(total_val)
        total_val = np.vstack((par, total_val))
        total_sigma = np.vstack((sigma, total_sigma))
        f = np.asarray([f_par] + f.tolist())
        if find_max:
            bst = np.argmax(f)
            return total_val[bst], total_sigma[bst], f[bst]
        else:
            bst = np.argmin(f)
            return total_val[bst], total_sigma[bst], f[bst]

    def __reproduction_greedy(self, par, sigma, f_par, find_max):  # --- new arg ---
        c1_values = self.__crossover_method_1(par)
        c1_sigma = self.__crossover_method_1(sigma)
        c2_values = self.__crossover_method_1(par)
        c2_sigma = self.__crossover_method_1(sigma)
        c3_values, c3_sigma = self.__mutation_self_adaptive(c1_values, c1_sigma)
        c4_values, c4_sigma = self.__mutation_self_adaptive(c2_values, c2_sigma)
        c5_values = self.__crossover_method_2(par)
        c5_sigma = self.__crossover_method_2(sigma)
        c6_values = self.__crossover_method_2(par)
        c6_sigma = self.__crossover_method_2(sigma)
        c7_values, c7_sigma = self.__mutation_self_adaptive(c5_values, c5_sigma)
        c8_values, c8_sigma = self.__mutation_self_adaptive(c6_values, c6_sigma)

        total_val = [c1_values, c2_values, c3_values, c4_values, c5_values, c6_values, c7_values, c8_values]
        total_sigma = [c1_sigma, c2_sigma, c3_sigma, c4_sigma, c5_sigma, c6_sigma, c7_sigma, c8_sigma]

        for i in range(0, 8):
            for j in range(0, len(c1_values)):
                if total_val[i][j] > self.upper_bound[j]:
                    total_val[i][j] = self.upper_bound[j]
                    total_sigma[i][j] *= 0.9
                elif total_val[i][j] < self.lower_bound[j]:
                    total_val[i][j] = self.lower_bound[j]
                    total_sigma[i][j] *= 0.9

        f = self.fitness_function(np.asarray(total_val))
        total_val = par + total_val
        total_sigma = sigma + total_sigma
        f = f_par + f.tolist()
        if find_max:
            bst = np.argmax(f)
            return total_val[bst], total_sigma[bst], f[bst]
        else:
            bst = np.argmin(f)
            return total_val[bst], total_sigma[bst], f[bst]

    def __mutation_1_n_z(self, x1, xs, beta):
        return x1 + beta * (xs[0] - xs[1])

    def __differential(self, par, n, find_max, beta=0.5):
        ind = np.random.choice(range(0, n), 3, replace=False)
        target = self.gen[ind[2]]
        unit = self.__mutation_1_n_z(target, self.gen[ind[0:2]], beta)
        child = self.__crossover_method_1([unit, par])
        for j in range(0, len(child)):
            if child[j] > self.upper_bound[j]:
                child[j] = self.upper_bound[j]
            elif child[j] < self.lower_bound[j]:
                child[j] = self.lower_bound[j]
        for j in range(0, len(child)):
            if unit[j] > self.upper_bound[j]:
                unit[j] = self.upper_bound[j]
            elif unit[j] < self.lower_bound[j]:
                unit[j] = self.lower_bound[j]
        total = np.asarray([child])
        f = self.fitness_function(total)
        if find_max:
            bst = np.argmax(f)
            return f[bst], total[bst]
        else:
            bst = np.argmin(f)
            return f[bst], total[bst]

    # greedy - proportional selection for mating - 4 cross 4 mut
    # differential
    def evolve(self, algorithm='greedy', max_iter=100, info=True, find_max=False,
                warm_start=False, init_pop=None):
        """Perform evolution with the given set of parameters
        Parameters
        -----------
        algorithm : string , default = 'greedy'
                   A string to denote the algorithm to be used for evolution.
                   Three algorithms are currently available: 'greedy', 'generic',
                   and 'self-adaptive'. Please see the example notebooks for a
                   run down.
        max_iter : int
                  The maximum number of iterations to run before terminating
        info : bool
              If True, print out information during the evolution process
        find_max : bool
                  If True, the algorithm will try to maximize the fitness
                  function given; else it will minimize.
        warm_start : bool
                    If True, the algorithm will use the last generation
                    from the previous generation instead of creating a
                    new initial population
        init_pop : int or None
                  If not None, the algorithm will create an initial population
                  size equal to `init_pop` and then will reduce down to
                  `gen_size` amount for evolution by choosing the best individuals.
                  For example, if `init_pop` is 100 and `gen_size` is 20, an initial
                  population of 100 individual will be created but only the best 20
                  will be carried over for evolution. This parameter is used to explore
                  the domain space and utilize the genetic algorithm to converge and
                  exploit the best set of individuals during the evolution process.
        """
        if not warm_start:
            self.mean_fit = []
            self.best_fit = []
            if init_pop is None:
                init_gen = np.empty(shape=(self.gen_size, self.num_variables))
                for i in range(0, self.num_variables):
                    init_gen[:, i] = np.random.uniform(self.lower_bound[i], self.upper_bound[i], self.gen_size)
                self.gen = np.copy(init_gen)

                if algorithm != 'differential':
                    init_sigma = np.empty(shape=(self.gen_size, self.num_variables))
                    for i in range(0, self.num_variables):
                        init_sigma[:, i] = np.random.uniform(0.01 * self.total_bound[i],
                                                             0.2 * self.total_bound[i], self.gen_size)
                    self.sigma = np.copy(init_sigma)
                fitness = self.fitness_function(self.gen)
            else:
                if info:
                    msg = "Starting Random Initial Population..."
                    print(msg)
                init_gen = np.empty(shape=(init_pop, self.num_variables))
                for i in range(0, self.num_variables):
                    init_gen[:, i] = np.random.uniform(self.lower_bound[i], self.upper_bound[i], init_pop)
                self.gen = np.copy(init_gen)

                if algorithm != 'differential':
                    init_sigma = np.empty(shape=(init_pop, self.num_variables))
                    for i in range(0, self.num_variables):
                        init_sigma[:, i] = np.random.uniform(0.01 * self.total_bound[i],
                                                             0.2 * self.total_bound[i], init_pop)
                    self.sigma = np.copy(init_sigma)

                fitness = self.fitness_function(self.gen, init_pop_print=True)
                if find_max:
                    bst = np.argsort(-fitness)[0:self.gen_size]
                else:
                    bst = np.argsort(fitness)[0:self.gen_size]
                fitness = fitness[bst]
                if info:
                    msg = "Random Initial Population Finished..."
                    print(msg)
                self.gen = self.gen[bst]
                if algorithm != 'differential':
                    self.sigma = self.sigma[bst]
                if info:
                    msg = "Starting Evolution Process on Best {} Individuals...".format(self.gen_size)
                    print(msg)
        else:
            fitness = self.fitness_function(self.gen)

        n, c = np.shape(self.gen)
        for k in range(0, max_iter):
            fit_mean = np.mean(fitness)
            if find_max:
                fit_best = np.max(fitness)
                best_index = np.argmax(fitness)
            else:
                fit_best = np.min(fitness)
                best_index = np.argmin(fitness)
            self.best_values.append(self.gen[best_index,])
            self.best_fit.append(fit_best)
            self.mean_fit.append(fit_mean)
            if info:
                msg = "GENERATION {}:\n" \
                      "  Best Fit: {}, Mean Fit: {}".format(k, fit_best, fit_mean)
                print(msg)

            if algorithm == 'differential':
                if find_max:
                    coef = fit_mean / fit_best
                else:
                    coef = fit_best / fit_mean
                if coef > 0.95:
                    beta = 1
                elif coef < 0.2:
                    beta = 0.2
                else:
                    beta = 0.55
                for i in range(0, n):
                    par = self.gen[i]
                    f, child = self.__differential(par, n, find_max,beta=beta)
                    if find_max:
                        if f > fitness[i]:
                            fitness[i] = f
                            self.gen[i] = child
                    else:
                        if f < fitness[i]:
                            fitness[i] = f
                            self.gen[i] = child
            elif algorithm == 'greedy':
                mates1 = np.random.choice(range(0, n), n, replace=False)
                mates2 = np.random.choice(range(0, n), n, replace=False)
                children = []
                children_sigma = []
                fits = []
                for i in range(0, n):
                    v, s, f = self.__reproduction_greedy([self.gen[mates1[i]], self.gen[mates2[i]]],
                                                         [self.sigma[mates1[i]], self.sigma[mates2[i]]],
                                                         [fitness[mates1[i]], fitness[mates2[i]]],
                                                         find_max)
                    children.append(v)
                    children_sigma.append(s)
                    fits.append(f)
                fitness = fits

            elif algorithm == 'self-adaptive':
                children = []
                children_sigma = []
                fits = []
                for i in range(0, n):
                    c_value, c_sigma, f = self.__reproduction_self_adapt(self.gen[i], self.sigma[i], fitness[i], find_max)
                    children.append(c_value)
                    children_sigma.append(c_sigma)
                    fits.append(f)
                fitness = fits

            if algorithm != 'differential':
                gen_next = np.asarray(children)
                sigma_next = np.asarray(children_sigma)

                self.gen = gen_next
                if algorithm == 'self-adaptive':
                    self.sigma = sigma_next

        if find_max:
            self.best_individual = self.gen[np.argmax(fitness)]
        else:
            self.best_individual = self.gen[np.argmin(fitness)]

    def plot(self, starting_gen=0):
        """Plots the best and mean fitness values after the evolution process.
        Parameters
        -----------
        starting_gen : int
                      The starting index for plotting.
        """
        x_range = range(starting_gen, len(self.best_fit))
        plt.plot(x_range, self.mean_fit[starting_gen:], label="Mean Fitness")
        plt.plot(x_range, self.best_fit[starting_gen:], label="Best Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Value")
        plt.suptitle("Mean and Best Fitness for Algorithm: ")
        plt.legend()
        plt.show()

class Saved_Weights:

    def __init__(self, weights, type, val_loss):
        self.type = type
        self.weights = weights
        self.val_loss = val_loss

def load_weights(model, save_dir):
    cnn_mod_count = 0
    deep_mod_count = 0
    for i in range(0, len(model.layers)):
        layer = model.layers[i]
        if isinstance(layer, Conv2D) or isinstance(layer, SeparableConv2D):
            sph = layer.name
            file = save_dir + "/conv2d/" + sph
            if os.path.isfile(file):
                pre_mod = pickle.load(open(file, "rb"))
                layer.set_weights(pre_mod.weights)
            cnn_mod_count += 1
        elif isinstance(layer, Flatten):
            deep_mod_count += 1
        elif isinstance(layer, Dense):
            sph = layer.name
            file = save_dir + "/dense/" + sph
            if os.path.isfile(file):
                pre_mod = pickle.load(open(file, "rb"))
                layer.set_weights(pre_mod.weights)
        elif isinstance(layer, BatchNormalization):
            sph = layer.name
            if deep_mod_count == 0:
                file = save_dir + "/batch_norm/" + sph
                if os.path.isfile(file):
                    pre_mod = pickle.load(open(file, "rb"))
                    layer.set_weights(pre_mod.weights)

def save_weights(model, val_loss, save_dir):
    cnn_mod_count = 0
    deep_mod_count = 0
    for layer in model.layers:
        if isinstance(layer, Conv2D) or isinstance(layer, SeparableConv2D):
            weights = layer.get_weights()
            sph = layer.name
            file = save_dir + "/conv2d/" + sph
            if os.path.isfile(file):
                pre_mod = pickle.load(open(file, "rb"))
                if pre_mod.val_loss > val_loss:
                    pickle.dump(Saved_Weights(weights=weights, type='conv2d', val_loss=val_loss), open(file, "wb"))
            else:
                pickle.dump(Saved_Weights(weights=weights, type='conv2d', val_loss=val_loss), open(file, "wb"))
            cnn_mod_count += 1
        elif isinstance(layer, Flatten):
            deep_mod_count += 1
        elif isinstance(layer, Dense):
            weights = layer.get_weights()
            sph = layer.name
            file = save_dir + "/dense/" + sph
            if os.path.isfile(file):
                pre_mod = pickle.load(open(file, "rb"))
                if pre_mod.val_loss > val_loss:
                    pickle.dump(Saved_Weights(weights=weights, type='dense', val_loss=val_loss), open(file, "wb"))
            else:
                pickle.dump(Saved_Weights(weights=weights, type='dense', val_loss=val_loss), open(file, "wb"))
        elif isinstance(layer, BatchNormalization):
            weights = layer.get_weights()
            sph = layer.name
            if deep_mod_count == 0:
                file = save_dir + "/batch_norm/" + sph
                if os.path.isfile(file):
                    pre_mod = pickle.load(open(file, "rb"))
                    if pre_mod.val_loss > val_loss:
                        pickle.dump(Saved_Weights(weights=weights, type='batch_norm', val_loss=val_loss),
                                    open(file, "wb"))
                else:
                    pickle.dump(Saved_Weights(weights=weights, type='batch_norm', val_loss=val_loss),
                                open(file, "wb"))

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

class InitialPopulation:

    def  __init__(self, fitness_function_phase1, cnn_min_blocks, cnn_max_blocks, dense_min_blocks, dense_max_blocks,
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
        self.fitness_function_phase1 = fitness_function_phase1
        self.init_pop_size = init_pop_size
        self.init_individuals = []
        self.init_fitness = []
        self.save_dir = save_dir
        self.time = 0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir + '/conv2d')
            os.makedirs(save_dir + '/batch_norm')
            os.makedirs(save_dir + '/dense')

    def fit(self):
        msg = "STARTING INITIAL POPULATION"
        print(msg)
        logging.info(msg)
        index = 0
        start = time.time()
        while index < self.init_pop_size:

            individual = Individual(cnn_min_blocks=self.cnn_min_blocks, cnn_max_blocks=self.cnn_max_blocks,
                                    dense_min_blocks=self.dense_min_blocks, dense_max_blocks=self.dense_max_blocks,
                 min_modules=self.min_modules, max_node=self.max_node, min_node=self.min_node, input_shape=self.input_shape,
                                    filters=self.filters, num_output=self.num_output, output_act=self.output_act)

            model = individual.build_model()
            load_weights(model=model, save_dir=self.save_dir)
            result = self.fitness_function_phase1(model)

            if result is None:
                msg = " MODEL ARCHITECTURE FAILED..."
                print(msg)
                logging.info(msg)
                continue

            f, v, t = result
            save_weights(model=model, save_dir=self.save_dir, val_loss=v)
            num_param = model.count_params()
            msg = " MODEL {} -> Val Acc: {}, Val Loss: {}, Time/Epoch: {}, num_param: {}".format(index,
                                                                                                f, v, t, num_param)
            print(msg)
            logging.info(msg)
            self.init_individuals.append(individual)
            self.init_fitness.append(f)
            index += 1

        finish = time.time()
        self.time = finish-start
        msg = " Time Elapsed: {} min".format(self.time / 60.0)
        print(msg)
        logging.info(msg)
        self.init_fitness = np.asarray(self.init_fitness)
        self.init_individuals = np.asarray(self.init_individuals)

class DifferentialEvolution:
    def __crossover_method_1(self, par):
        return np.mean(par, axis=0)

    def __crossover_method_2(self, par):
        child = np.copy(par[0])
        n = np.shape(par[0])[0]
        random_nums = np.random.randint(low=0, high=len(par), size=n)
        for j in range(0, n):
            child[j] = par[random_nums[j]][j]
        return child

    def __crossover_method_3(self, par):
        pass

    def __mutation_1_n_z(self, x1, xs1, xs2, beta):
        return x1 + beta * (xs1 - xs2)

    def __differential(self, par, beta=0.55):
        ind = np.random.choice(range(0, self.gen_size), 3, replace=False)
        target = self.gen[ind[2]]
        child = Individual(cnn_min_blocks=self.initial_population.cnn_min_blocks, cnn_max_blocks=self.initial_population.cnn_max_blocks,
                                    dense_min_blocks=self.initial_population.dense_min_blocks, dense_max_blocks=self.initial_population.dense_max_blocks,
                 min_modules=self.initial_population.min_modules, max_node=self.initial_population.max_node, min_node=self.initial_population.min_node, input_shape=self.initial_population.input_shape,
                                    filters=self.initial_population.filters, num_output=self.initial_population.num_output, output_act=self.initial_population.output_act)

        for i in range(0, child.num_modules):
            for j in range(0, child.cnn_modules[i].max_blocks):
                child.cnn_modules[i].cnn_blocks[j].chromosome = self.__mutation_1_n_z(
                        target.cnn_modules[i].cnn_blocks[j].chromosome,
                      self.gen[ind[0]].cnn_modules[i].cnn_blocks[j].chromosome,
                      self.gen[ind[1]].cnn_modules[i].cnn_blocks[j].chromosome,
                      beta)

                child.cnn_modules[i].cnn_blocks[j].beliefs_probs = self.__mutation_1_n_z(
                    target.cnn_modules[i].cnn_blocks[j].beliefs_probs,
                    self.gen[ind[0]].cnn_modules[i].cnn_blocks[j].beliefs_probs,
                    self.gen[ind[1]].cnn_modules[i].cnn_blocks[j].beliefs_probs,
                    beta)

                child.cnn_modules[i].cnn_blocks[j].beliefs_act_weights = self.__mutation_1_n_z(
                    target.cnn_modules[i].cnn_blocks[j].beliefs_act_weights,
                    self.gen[ind[0]].cnn_modules[i].cnn_blocks[j].beliefs_act_weights,
                    self.gen[ind[1]].cnn_modules[i].cnn_blocks[j].beliefs_act_weights,
                    beta)

                if self.cross_method == 1:
                    child.cnn_modules[i].cnn_blocks[j].chromosome = self.__crossover_method_1([
                        child.cnn_modules[i].cnn_blocks[j].chromosome,
                        par.cnn_modules[i].cnn_blocks[j].chromosome
                    ])

                    child.cnn_modules[i].cnn_blocks[j].beliefs_probs = self.__crossover_method_1([
                        child.cnn_modules[i].cnn_blocks[j].beliefs_probs,
                        par.cnn_modules[i].cnn_blocks[j].beliefs_probs
                    ])

                    child.cnn_modules[i].cnn_blocks[j].beliefs_act_weights = self.__crossover_method_1([
                        child.cnn_modules[i].cnn_blocks[j].beliefs_act_weights,
                        par.cnn_modules[i].cnn_blocks[j].beliefs_act_weights
                    ])

                child.cnn_modules[i].cnn_blocks[j].update_beliefs()

        for j in range(0, child.dense_module.max_blocks):
            child.dense_module.dense_blocks[j].chromosome = self.__mutation_1_n_z(
                target.dense_module.dense_blocks[j].chromosome,
                self.gen[ind[0]].dense_module.dense_blocks[j].chromosome,
                self.gen[ind[1]].dense_module.dense_blocks[j].chromosome,
                beta
            )

            child.dense_module.dense_blocks[j].dense_beliefs = self.__mutation_1_n_z(
                target.dense_module.dense_blocks[j].dense_beliefs,
                self.gen[ind[0]].dense_module.dense_blocks[j].dense_beliefs,
                self.gen[ind[1]].dense_module.dense_blocks[j].dense_beliefs,
                beta
            )

            child.dense_module.dense_blocks[j].dense_act_type_weights = self.__mutation_1_n_z(
                target.dense_module.dense_blocks[j].dense_act_type_weights,
                self.gen[ind[0]].dense_module.dense_blocks[j].dense_act_type_weights,
                self.gen[ind[1]].dense_module.dense_blocks[j].dense_act_type_weights,
                beta
            )

            if self.cross_method == 1:
                child.dense_module.dense_blocks[j].chromosome = self.__crossover_method_1([
                    child.dense_module.dense_blocks[j].chromosome,
                    par.dense_module.dense_blocks[j].chromosome
                ])
                child.dense_module.dense_blocks[j].dense_beliefs = self.__crossover_method_1([
                    child.dense_module.dense_blocks[j].dense_beliefs,
                    par.dense_module.dense_blocks[j].dense_beliefs
                ])
                child.dense_module.dense_blocks[j].dense_act_type_weights = self.__crossover_method_1([
                    child.dense_module.dense_blocks[j].dense_act_type_weights,
                    par.dense_module.dense_blocks[j].dense_act_type_weights
                ])

            child.dense_module.dense_blocks[j].update_beliefs()

        model = child.build_model()
        load_weights(model=model, save_dir=self.save_dir)
        result = self.initial_population.fitness_function_phase1(model)
        if result is not None:
            save_weights(model=model, val_loss=result[1], save_dir=self.save_dir)
        return result, child

    def __init__(self, cross_method, initial_population, gen_size, max_iter, save_dir, fitness_function_phase2,
                 fitness_function_phase3, fitness_function_phase4):

        self.initial_population = initial_population
        self.cross_method = cross_method
        self.gen_size = gen_size
        self.max_iter = max_iter
        self.save_dir = save_dir
        self.fitness_function_phase2 = fitness_function_phase2
        self.fitness_function_phase3 = fitness_function_phase3
        self.fitness_function_phase4 = fitness_function_phase4
        self.gen = []
        self.fitness = []
        self.best_individuals = []
        self.best_fit = []
        self.mean_fit = []
        self.prev_individuals = []
        self.gen_phase3 = []
        self.phase3_best_individuals = []
        self.phase3_best_fit = []
        self.phase3_best_individual = []
        self.phase3_mean_fit = []
        self.phase4_preds = []
        self.n_bounds = [1, 8]
        self.m_bounds = [0, 30]
        self.phase4_coef = []
        self.phase1_time = 0
        self.phase2_time = 0
        self.phase3_time = 0
        self.phase4_time = 0
        self.best = []

    def evolve_phase1(self):

        msg = "STARTING PHASE 1"
        print(msg)
        logging.info(msg)
        start = time.time()
        bst = np.argsort(-self.initial_population.init_fitness)[0:self.gen_size]

        self.gen = self.initial_population.init_individuals[bst]
        self.fitness = np.copy(self.initial_population.init_fitness[bst])

        for k in range(0, self.max_iter):
            fit_mean = np.mean(self.fitness)
            fit_best = np.max(self.fitness)
            best_index = np.argmax(self.fitness)
            self.best_individuals.append(self.gen[best_index])
            self.best_fit.append(fit_best)
            self.mean_fit.append(fit_mean)
            msg = " GENERATION {}:\n" \
                  "   Best Fit: {}, Mean Fit: {}".format(k, fit_best, fit_mean)
            print(msg)
            logging.info(msg)

            coef = fit_mean / fit_best
            if coef > 0.95:
                beta = 1
            elif coef < 0.2:
                beta = 0.2
            else:
                beta = 0.55

            for i in range(0, self.gen_size):
                par = self.gen[i]
                repeat = 0
                f = -1
                while True:
                    result, child = self.__differential(par, beta=beta)
                    if result is None:
                        repeat += 1
                        if repeat == 3:
                            msg = " -> FAILED EVOLVE MODEL"
                            print(msg)
                            logging.info(msg)
                            break
                    else:
                        f, v, t = result
                        msg = " -> Val Acc: {}, Val Loss: {}, Time/Epoch: {}, num_param: {}".format(f, v, t, child.num_param)
                        print(msg)
                        logging.info(msg)
                        break

                if f > self.fitness[i]:
                    self.fitness[i] = f
                    self.prev_individuals.append(self.gen[i])
                    child.gen = k
                    self.gen[i] = child
                else:
                    self.gen[i].age += 1

        finish = time.time()
        self.phase1_time = finish - start
        msg = " Time Elapsed: {} min".format(self.phase1_time / 60.0)
        print(msg)
        logging.info(msg)

    def evolve_phase2(self, top_gen):
        msg = "STARTING PHASE 2"
        print(msg)
        logging.info(msg)
        start = time.time()
        bst = np.argsort(-self.fitness)[0:top_gen]

        self.best = self.gen[bst]

        for i in range(0, top_gen):
            child = self.best[i]

            model = child.build_model()
            load_weights(model=model, save_dir=self.save_dir)
            result = self.fitness_function_phase2(model)
            model.save(self.save_dir + "/phase2_model_{}".format(i))

            f, v, t = result
            msg = " -> Val Acc: {}, Val Loss: {}, Time/Epoch: {}, num_param: {}".format(f, v, t, child.num_param)
            print(msg)
            logging.info(msg)

        finish = time.time()
        self.phase2_time = finish - start
        msg = " Time Elapsed: {} min".format(self.phase2_time/60.0)
        print(msg)
        logging.info(msg)

    def __differential_phase3(self, par, target, ind1, ind2, model, beta):
        unit = self.__mutation_1_n_z(target, ind1, ind2, beta)
        child = self.__crossover_method_1([unit, par])
        if child[0] < self.n_bounds[0]:
            child[0] = self.n_bounds[0]
        elif child[0] > self.n_bounds[1]:
            child[0] = self.n_bounds[1]

        if child[1] < self.m_bounds[0]:
            child[1] = self.m_bounds[0]
        elif child[1] > self.m_bounds[1]:
            child[1] = self.m_bounds[1]

        n = int(np.rint(child[0]))
        m = int(np.rint(child[1]))
        result = self.fitness_function_phase3(model, n, m)
        f, v, t = result
        msg = " -> N: {} - M: {}, Val Acc: {}, Val Loss: {}, Time/Epoch: {}".format(n, m, f, v, t)
        print(msg)
        logging.info(msg)

        return f, child

    def evolve_phase3(self, top_gen, gen_size, init_size, max_iter):

        msg = "STARTING PHASE 3"
        print(msg)
        logging.info(msg)
        start = time.time()
        self.gen_phase3 = np.empty(shape=(top_gen, init_size, 2))
        self.phase3_best_individuals = np.empty(shape=(top_gen, max_iter, 2))
        self.phase3_best_fit = np.empty(shape=(top_gen, max_iter))
        self.phase3_best_individual = np.empty(shape=(top_gen, 2))
        self.phase3_mean_fit = np.empty(shape=(top_gen, max_iter))
        self.gen_phase3[:, :, 0] = np.random.uniform(self.n_bounds[0], self.n_bounds[1], top_gen * init_size).reshape(
            top_gen, init_size)
        self.gen_phase3[:, :, 1] = np.random.uniform(self.m_bounds[0], self.m_bounds[1], top_gen * init_size).reshape(
            top_gen, init_size)

        new_gen = []

        for i in range(0, top_gen):
            msg = " MODEL {}".format(i)
            print(msg)
            logging.info(msg)
            fitness = []
            for k in range(0, init_size):
                model = tf.keras.models.load_model(self.save_dir + "/phase2_model_{}".format(i))
                n = int(np.rint(self.gen_phase3[i][k][0]))
                m = int(np.rint(self.gen_phase3[i][k][1]))
                result = self.fitness_function_phase3(model=model, n=n, m=m)
                fitness.append(result[0])
                if np.argmax(fitness) == k:
                    model.save(self.save_dir + "/phase4_model_{}".format(i))
            fitness = np.asarray(fitness)
            bst = np.argsort(-fitness)[0:gen_size]

            gen = self.gen_phase3[i][bst]
            fitness = fitness[bst]
            for k in range(0, max_iter):
                fit_mean = np.mean(fitness)
                fit_best = np.max(fitness)
                best_index = np.argmax(fitness)
                self.phase3_best_individuals[i][k] = self.gen_phase3[i][best_index]
                self.phase3_best_fit[i][k] = fit_best
                self.phase3_mean_fit[i][k] = fit_mean
                msg = "  GENERATION {}:\n" \
                      "    Best Fit: {}, Mean Fit: {}".format(k, fit_best, fit_mean)
                print(msg)
                logging.info(msg)
                coef = fit_mean / fit_best
                if coef > 0.95:
                    beta = 1
                elif coef < 0.2:
                    beta = 0.2
                else:
                    beta = 0.55

                for j in range(0, gen_size):
                    model = tf.keras.models.load_model(self.save_dir + "/phase4_model_{}".format(i))
                    par = gen[j]
                    ind = np.random.choice(range(0, gen_size), 3, replace=False)
                    f, child = self.__differential_phase3(par, gen[ind[2]], gen[ind[0]], gen[ind[1]],
                                                          model, beta)
                    if f > fitness[j]:
                        fitness[j] = f
                        gen[j] = child
                        if np.argmax(fitness) == j:
                            model.save(self.save_dir + "/phase4_model_{}".format(i))

            new_gen.append(gen)
            self.phase3_best_individual[i] = gen[np.argmax(fitness)]

        self.gen_phase3 = np.asarray(new_gen)
        finish = time.time()
        self.phase3_time = finish - start
        msg = " Time Elapsed: {} min".format(self.phase3_time/60.0)
        print(msg)
        logging.info(msg)

    def predict(self, x):
        model = tf.keras.models.load_model(self.save_dir + "/phase4_model_{}".format(0))
        preds = self.phase4_coef[0] * model.predict(x)
        for i in range(0, len(self.phase3_best_individual)):
            model = tf.keras.models.load_model(self.save_dir + "/phase4_model_{}".format(i))
            preds += self.phase4_coef[i] * model.predict(x)
        return np.argmax(preds, axis=1)

    def evolve_phase4(self, x_validation, y_validation):
        msg = "STARTING PHASE 4"
        print(msg)
        logging.info(msg)
        start = time.time()
        for i in range(0, len(self.phase3_best_individual)):
            msg = " MODEL {}".format(i)
            print(msg)
            logging.info(msg)
            model = tf.keras.models.load_model(self.save_dir + "/phase4_model_{}".format(i))
            temp = self.fitness_function_phase4(model)
            acc = accuracy_score(np.argmax(y_validation, axis=1), np.argmax(temp, axis=1))
            msg = "  - Val Acc: {}".format(acc)
            print(msg)
            logging.info(msg)
            self.phase4_preds.append(temp)
            model.save(self.save_dir + "/phase4_model_{}".format(i))

        self.phase4_preds = np.asarray(self.phase4_preds)

        # number of dimensions
        d = len(self.phase3_best_individual)

        # define bounds
        lower_bound = [0] * d
        upper_bound = [1] * d

        n = len(self.phase4_preds[0])
        t = self.phase4_preds

        def fitness_function_test(x):
            batch_size = 1000
            if len(x.shape) == 1:
                acc = []
                for index in range(0, n, batch_size):
                    batch = t[:, index:min(index + batch_size, n)]
                    sm = batch[0] * x[0]
                    for i in range(1, len(x)):
                        sm += batch[i] * x[i]
                    acc.append(accuracy_score(np.argmax(y_validation[index:min(index + batch_size, n)], axis=1),
                                              np.argmax(sm, axis=1)))
                return np.mean(acc)
            else:
                acc2 = []
                for row in x:
                    acc = []
                    for index in range(0, n, batch_size):
                        batch = t[:, index:min(index + batch_size, n)]
                        sm = batch[0] * row[0]
                        for i in range(1, len(row)):
                            sm += batch[i] * row[i]
                        acc.append(accuracy_score(np.argmax(y_validation[index:min(index + batch_size, n)], axis=1),
                                                  np.argmax(sm, axis=1)))
                    acc2.append(np.mean(acc))
                return np.asarray(acc2)

        fits = []
        best = []
        for i in range(0, 5):  # run algo 5 times
            # define number of solutions per generation
            gen_size = 100
            algorithm = GenericUnconstrainedProblem(fitness_function=fitness_function_test,
                                                       upper_bound=upper_bound, lower_bound=lower_bound,
                                                       gen_size=gen_size)
            algorithm.evolve(max_iter=25, algorithm='greedy', find_max=True, info=False)
            self.phase4_coef = algorithm.best_individual
            preds = self.predict(x_validation)
            val_acc = accuracy_score(np.argmax(y_validation, axis=1), preds)
            fits.append(val_acc)
            print(" Iteration {} - Val Acc: {}".format(i, val_acc))
            best.append(algorithm.best_individual)

        self.phase4_coef = best[np.argmax(fits)]
        msg = " Best Val Acc: {} -- Coef: {}".format(np.max(fits), self.phase4_coef)
        print(msg)
        logging.info(msg)
        finish = time.time()
        self.phase4_time = finish - start
        msg = " Time Elapsed: {} min".format(self.phase4_time/60.0)
        print(msg)
        logging.info(msg)

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

def fitness_function_phase1(model, epochs=5):
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

    f = np.nanmax(history.history['val_accuracy'])
    if np.nanmax(history.history['accuracy']) <= min_acc:
        return None
    v = np.nanmin(history.history['val_loss'])
    t = np.median(timeCallback.times)
    return f, v, t

def fitness_function_phase2(model, epochs=3):
    opt = tf.keras.optimizers.Adam(clipnorm=1.0)
    timeCallback = TimeHistory()
    term = TerminateOnNaN()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    callback = [EarlyStopping(monitor='loss', patience=25, restore_best_weights=True),
                EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True),
                EarlyStopping(monitor='val_loss', patience=75, restore_best_weights=True),
                EarlyStopping(monitor='accuracy', patience=15, restore_best_weights=True),
                timeCallback,
                term]

    history = model.fit(x_train[train_ind_full], y_train[train_ind_full], batch_size=batch_size, epochs=epochs,
            verbose=verbose, callbacks=callback, validation_data=(x_train[val_ind_full], y_train[val_ind_full]))

    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmin(history.history['val_loss'])
    t = np.median(timeCallback.times)
    return f, v, t

def fitness_function_phase3(model, n, m, epochs=3):
    opt = tf.keras.optimizers.Adam(clipnorm=1.0)
    timeCallback = TimeHistory()
    term = TerminateOnNaN()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    callback = [EarlyStopping(monitor='loss', patience=25, restore_best_weights=True),
                EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True),
                EarlyStopping(monitor='val_loss', patience=75, restore_best_weights=True),
                EarlyStopping(monitor='accuracy', patience=15, restore_best_weights=True),
                timeCallback,
                term]

    rand_aug = iaa.RandAugment(n=n, m=m)

    def augment(images):
        # Input to `augment()` is a TensorFlow tensor which
        # is not supported by `imgaug`. This is why we first
        # convert it to its `numpy` variant.
        images = tf.cast(images, tf.uint8)
        return rand_aug(images=images.numpy())

    aug_train = augment(x_train[train_ind_full])  #train_ind_full

    history = model.fit(aug_train, y_train[train_ind_full], batch_size=batch_size, epochs=epochs,
            verbose=verbose, callbacks=callback, validation_data=(x_train[val_ind_full], y_train[val_ind_full]))

    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmin(history.history['val_loss'])
    t = np.median(timeCallback.times)
    return f, v, t

def fitness_function_phase4(model):

    return model.predict(x_train[val_ind_full])

def create_parser():
    '''
    Create a command line parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='Differential Evolution')
    parser.add_argument('--cross_method', type=int, default=1, help='Crossover Method')
    parser.add_argument('--gen_size', type=int, default=10, help='Generation Size for Evolution')
    parser.add_argument('--max_iter', type=int, default=3, help='Number of Iterations for Evolution')
    parser.add_argument('--top_gen', type=int, default=5, help='Number of Best Individuals for Phase2')

    parser.add_argument('--phase3_gen_size', type=int, default=3, help='Phase3 Generation Size for Evolution')
    parser.add_argument('--phase3_max_iter', type=int, default=3, help='Phase3 Number of Iterations for Evolution')
    parser.add_argument('--phase3_top_gen', type=int, default=3, help='Phase3 Number of Best Individuals')
    parser.add_argument('--phase3_init_size', type=int, default=5, help='Phase3 Initial Population Size')

    parser.add_argument('--cnn_min_blocks', type=int, default=2, help='Minimum # of blocks per CNN Module')
    parser.add_argument('--cnn_max_blocks', type=int, default=4, help='Maximum # of blocks per CNN Module')
    parser.add_argument('--dense_max_blocks', type=int, default=2, help='Maximum # of blocks per Dense Module')
    parser.add_argument('--dense_min_blocks', type=int, default=0, help='Minimum # of blocks per Dense Module')
    parser.add_argument('--min_nodes', type=int, default=100, help='Min Number of hidden units')
    parser.add_argument('--max_nodes', type=int, default=1000, help='Max Number of hidden units')
    parser.add_argument('--init_pop_size', type=int, default=20, help='Initial Population Size')
    parser.add_argument('--num_output', type=int, default=10, help='Number of output units')
    parser.add_argument('--output_act', type=str, default="softmax", help='Number of hidden units')
    parser.add_argument('--logs_file', type=str, default='ablation_logs989.log', help='Output File For Logging')
    parser.add_argument('--save_dir', type=str, default='differential_evolution_weights989', help='Save Directory for saving Model weights')
    parser.add_argument('--algo_save_file', type=str, default='differential_evolution_algo989',
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
    cross_method = args.cross_method
    gen_size = args.gen_size
    max_iter = args.max_iter
    save_dir = args.save_dir
    logs_file = args.logs_file
    top_gen = args.top_gen
    phase3_top_gen = args.phase3_top_gen
    phase3_max_iter = args.phase3_max_iter
    phase3_gen_size = args.phase3_gen_size
    phase3_init_size = args.phase3_init_size
    algo_save_file = args.algo_save_file
    logging.basicConfig(filename=logs_file, level=logging.DEBUG)

    start = time.time()
    msg = "--- Starting Differential Evolution ---"
    logging.info(msg)
    print(msg)

    init_pop = InitialPopulation(fitness_function_phase1=fitness_function_phase1, dense_max_blocks=dense_max_blocks,
                                 cnn_min_blocks=cnn_min_blocks, cnn_max_blocks=cnn_max_blocks, min_modules=min_module,
                                 max_node=max_nodes, min_node=min_nodes, input_shape=input_shape, num_output=num_output,
                                 output_act=output_act, filters=filters, init_pop_size=init_pop_size,
                                 dense_min_blocks=dense_min_blocks, save_dir=save_dir)

    init_pop.fit()

    algo = DifferentialEvolution(initial_population=init_pop, cross_method=cross_method, gen_size=gen_size,
                                 fitness_function_phase2=fitness_function_phase2, fitness_function_phase3=fitness_function_phase3,
                                 fitness_function_phase4=fitness_function_phase4, max_iter=max_iter, save_dir=save_dir)

    algo.evolve_phase1()
    pickle.dump(algo, open(save_dir+"/"+algo_save_file + "_phase1", "wb"))
    algo.evolve_phase2(top_gen=top_gen)
    pickle.dump(algo, open(save_dir+"/"+algo_save_file + "_phase2", "wb"))
    algo.evolve_phase3(top_gen=phase3_top_gen, max_iter=phase3_max_iter, gen_size=phase3_gen_size,
                       init_size=phase3_init_size)
    pickle.dump(algo, open(save_dir+"/"+algo_save_file + "_phase3", "wb"))
    algo.evolve_phase4(x_validation=x_train[val_ind_full], y_validation=y_train[val_ind_full])

    finish = time.time()

    msg = "--- ENDING Differential Evolution ---"
    print(msg)
    logging.info(msg)
    msg = "--- Total Time Taken: {} min ---".format((finish-start)/60.0)
    logging.info(msg)
    print(msg)
    pickle.dump(algo, open(save_dir+"/"+algo_save_file+"_FINAL", "wb"))
