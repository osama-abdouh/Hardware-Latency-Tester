import os
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from components.colors import colors
from components.diagnosis import diagnosis
from components.neural_network import neural_network
from components.search_space import search_space
from components.tuning_rules import tuning_rules
from components.tuning_rules_symbolic import tuning_rules_symbolic
from components.neural_sym_bridge import NeuralSymbolicBridge
from components.lfi_integration import LfiIntegration
from components.storing_experience import StoringExperience
from components.improvement_checker import ImprovementChecker
from components.integral import integrals
from shutil import copyfile

from modules.module import module

import config as cfg

class controller:
    """
    The controller class manages the training and tuning of the neural network,
    interfacing with the underlying modules, identifying possible problems affecting
    the architecture and how to solve them during iterations.
    """
    def __init__(self, X_train, Y_train, X_test, Y_test, n_classes):
        """
        All attributes for managing the training of the neural network are initialized,
        as well as auxiliary classes such as the one for interfacing with the symbolic part
        and the one for storing the training progress on DB.
        """
        # self.nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.n_classes = n_classes
        self.ss = search_space()
        self.space = self.ss.search_sp()
        self.tr = tuning_rules_symbolic(self.space, self.ss, self)
        self.nsb = NeuralSymbolicBridge() # problems e initial_facts
        self.db = StoringExperience()
        self.db.create_db()
        self.lfi = LfiIntegration(self.db)
        self.symbolic_tuning = []
        self.symbolic_diagnosis = []
        self.issues = []
        self.weight = 0.6
        #self.epsilon = 0.33
        self.new = None
        self.new_fc = None
        self.new_conv = None
        self.rem_conv = None
        self.rem_fc = None
        self.da = None
        self.model = None
        self.params = None
        self.iter = 0
        self.lacc = 0.15
        self.hloss = 1.2
        self.levels = [7, 10, 13]
        self.imp_checker = ImprovementChecker(self.db, self.lfi)
        self.modules = module(cfg.MOD_LIST)
        self.best_score = 0

    # The following methods are used to determine actions to be applied to the network structure,
    # for example addition or removal of convolutions and dense layers

    def set_case(self, new):
        """
        indicates if batch norm needs to be added
        """
        self.new = new

    def add_fc_layer(self, new_fc, c):
        """
        indicates if one or more dense layers needs to be added, with 'c' the number of dense layers
        """
        self.new_fc = [new_fc, c]

    def add_conv_section(self, new_conv, c):
        """
        indicates if one or more convolutional layers needs to be added, with 'c' the number of conv layers
        """
        self.new_conv = [new_conv, c]
        
    def remove_conv_section(self, rem_conv):
        """
        indicates, based on the value of the boolean 'rem_conv', if a convolutional layer needs to be removed
        """
        self.rem_conv = rem_conv

    def remove_fully_connected(self, rem_fc):
        """
        indicates, based on the value of the boolean 'rem_fc', if a dense layer needs to be removed
        """
        self.rem_fc = rem_fc

    def set_data_augmentation(self, da):
        """
        indicates, based on the value of the boolean 'da', if data augmentation is necessary
        """
        self.da = da

    def smooth(self, scalars):
        """
        This function allows the smoothing of values in a list of values,
        weighing the last value and the next one during the iterations.
        :param scalars: scalars list to be smoothed (acc or loss history)
        :return: list of smoothed values
        """
        # init variables, last will be the first el of the list,
        # smoothed is initialized as an empty list
        last = scalars[0]
        smoothed = list()

        # iterate over each value
        for point in scalars:
            # Calculate smoothed value
            smoothed_val = last * self.weight + (1 - self.weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def manage_configuration(self):
        """
        this function calls, if initially loaded, the function in the 'energy module' for managing the power consumed,
        searching for the best available configuration.
        """
        energy_name = "energy_module"

        # if the module has been loaded
        if energy_name in self.modules.modules_name:
            # get the index from the loaded modules and, once the object is obtained, call the function
            index = self.modules.modules_name.index(energy_name)
            self.modules.modules_obj[index].fix_configuration()

    def training(self, params):
        """
        Training and tasting the neural network
        training(Train, Labels_train, Test, Label_test)
        :return: model and training history self.nn = neural_network(X_train, Y_train, X_test, Y_test, n_classes)
        """
        self.params = params

        print(colors.OKBLUE, "|  --> START TRAINING\n", colors.ENDC)
        K.clear_session()
        self.nn = neural_network(self.X_train, self.Y_train, self.X_test, self.Y_test, self.n_classes, self.best_score)
        self.score, self.history, self.model, self.best_score = self.nn.training(params, self.new, self.new_fc, self.new_conv, self.rem_conv, self.rem_fc, self.da,
                                                                self.space)

        # update state of modules
        # each module will take the necessary args internally
        self.modules.state(self.score[0], self.score[1], self.model)

        # print modules informations
        self.modules.print()
        
        # log module values of training
        self.modules.log()

        # increase the number of iterations
        self.new_fc = False
        self.rem_conv = False
        self.rem_fc = False
        self.iter += 1

        # if no module has been loaded or is incorrect for the symbolic part
        # or all the weights of the loaded modules are zero
        # accuracy will be the value to be optimized
        # otherwise return the finale function value to be optimized
        if (len(self.modules.modules_obj) == 0) or not self.modules.all_zeros_weights() or not self.modules.ready():
            return -self.score[1]
        else:
            _, _, opt_value = self.modules.optimiziation()
            return opt_value

    def diagnosis(self):
        """
        method for diagnose possible issue like overfitting
        :return: call to tuning method or hp_space, model and accuracy(*-1)
        """
        print(colors.CYAN, "| START SYMBOLIC DIAGNOSIS ----------------------------------  |\n", colors.ENDC)
        diagnosis_logs = open("{}/algorithm_logs/diagnosis_symbolic_logs.txt".format(cfg.NAME_EXP), "a")
        tuning_logs = open("{}/algorithm_logs/tuning_symbolic_logs.txt".format(cfg.NAME_EXP), "a")

        # check if there has been an improvement from the last iteration 
        # also saves loss and accuracy values in the DB
        improv = self.imp_checker.checker(self.score[1], self.score[0])
        self.db.insert_ranking(self.score[1], self.score[0])

        # integral of loss history, useful in the symbolic part
        int_loss, int_slope = integrals(self.history['val_loss'])

        # at specific epochs, change  the threshold values for low accuracy and high loss detection
        for level in self.levels:
            if self.iter == level:
                self.lacc = self.lacc/2 + 0.05
                self.hloss = self.hloss/2 + 0.15

        # base facts list
        facts_list_module = [self.history['loss'], self.smooth(self.history['loss']),
             self.history['accuracy'], self.smooth(self.history['accuracy']),
             self.history['val_loss'], self.history['val_accuracy'], int_loss, int_slope, self.lacc, self.hloss]

        # add facts values from loaded modules
        facts_list_module += self.modules.values().values()

        # add facts and problems to NeuralSymbolicBridge
        # and create dynamic prolog file contains a list of possible problems
        # only during first diagnosis iteration
        if self.iter == 1:
            self.rules, self.actions, self.problems = self.modules.get_rules()

            for module, no_err in zip(self.modules.modules_obj, self.modules.modules_ready):
                # if there are no errors in the module, dynamically add facts and problems to the symbolic part
                if no_err:
                    self.nsb.initial_facts += module.facts
                    self.nsb.problems += module.problems

            # create a file containing all the logical rules for detecting problems in the network
            self.nsb.build_sym_prob(self.problems)

        # if there's data on improvement during training
        # analyse the problems in the neural network, as well as possible solutions,
        # and modify the probability with which these can be applied
        if improv is not None:
            _, lfi_problem = self.lfi.learning(improv, self.symbolic_tuning, self.symbolic_diagnosis, self.actions)
            sy_model = lfi_problem.get_model()
            self.nsb.edit_probs(sy_model)

        # analysing the neurla network through rule-based reasoning,
        # determining possible anomalies and solutions to these problems
        self.symbolic_tuning, self.symbolic_diagnosis = self.nsb.symbolic_reasoning(
            facts_list_module, diagnosis_logs, tuning_logs, self.rules)

        # close log files in which previous informations are stored
        diagnosis_logs.close()
        tuning_logs.close()
        
        # print(self.symbolic_tuning)
        # print(self.symbolic_diagnosis)
        for p in self.symbolic_diagnosis:
            print("I've found a problem: ", p)
        
        for s in self.symbolic_tuning:
            print("I've found a solution: ", s)

        print(colors.CYAN, "| END SYMBOLIC DIAGNOSIS   ----------------------------------  |\n", colors.ENDC)

        # if the network has anomalies, try to correct them through tuning operations,
        # returning the new hyper-parameter space at the end
        if self.symbolic_tuning:
            self.space, to_optimize, self.model = self.tuning()
            return self.space, to_optimize
        else:
            return self.space, -self.score[1]

    def tuning(self):
        """
        tuning the hyper-parameter space or add new hyper-parameters
        :return: new hp_space, new_model and accuracy(*-1) for the Bayesian Optimization
        """
        print(colors.FAIL, "| START SYMBOLIC TUNING    ----------------------------------  |\n", colors.ENDC)
        # tuning_logs = open("algorithm_logs/tuning_logs.txt", "a")
        # new_space, self.model = self.tr.repair(self, self.symbolic_tuning, tuning_logs, self.model, self.params)
        new_space, self.model = self.tr.repair(self.symbolic_tuning, self.symbolic_diagnosis, self.model, self.params)
        # tuning_logs.close()
        self.issues = []
        print(colors.FAIL, "| END SYMBOLIC TUNING      ----------------------------------  |\n", colors.ENDC)

        return new_space, -self.score[1], self.model

    def plotting_obj_function(self):
        """
        plot graphs from each loaded module
        """
        self.modules.plot()

    def save_experience(self):
        """
        the function saves a database containing the progress of the last training of the neural network
        """
        # separate the path from the database extension on which the data are stored
        # and add the name of the model to identify the db associated more easily
        db_split = os.path.splitext(self.db.db_name)
        db_dir = (db_split[0] + "-{}" + db_split[1]).format(self.nn.last_model_id)
        try:
            copyfile(self.db.db_name, db_dir)
        except:
            print(colors.FAIL, "|  -------------- FAILED TO SAVE DB -------------  |\n", colors.ENDC)
