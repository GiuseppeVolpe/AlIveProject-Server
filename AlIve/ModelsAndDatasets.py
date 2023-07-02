import os
import pickle
import shutil
import re
from datetime import datetime
from abc import ABC, abstractmethod

from tqdm import tqdm
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from sklearn.preprocessing import LabelBinarizer

from Utils import *

import matplotlib.pyplot as plt

from treelib import Node, Tree

#RANDOM INIT
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
#END OF RANDOM INIT

class SentenceLevelClassificationData:

    def __init__(self, 
                 train:pd.DataFrame, valid:pd.DataFrame, test:pd.DataFrame, 
                 text_column_name:str, target_column_name:str):
        
        self.__binarizer = None

        try:
            self.__trainfeatures, self.__traintargets = self.__encode(train, text_column_name, target_column_name)
            self.__validfeatures, self.__validtargets = self.__encode(valid, text_column_name, target_column_name)
            self.__testfeatures, self.__testtargets = self.__encode(test, text_column_name, target_column_name)
        except Exception as ex:
            raise Exception("Could not create dataset! {}".format(ex))
    
    def __encode(self, df:pd.DataFrame, text_column_name:str, target_column_name:str):

        if text_column_name not in df.columns or target_column_name not in df.columns:
            raise Exception("Invalid dataframe for dataset creation")
        
        df_copy = df.copy()

        df_features = df_copy.pop(text_column_name).values
        df_targets = df_copy.pop(target_column_name)

        binarizer = LabelBinarizer()
        encoded_df_targets = binarizer.fit_transform(df_targets.values.astype('str'))
        
        if self.__binarizer == None:
            self.__binarizer = binarizer

        return df_features, encoded_df_targets
    
    def get_train(self):
        return self.__trainfeatures, self.__traintargets
    
    def get_valid(self):
        return self.__validfeatures, self.__validtargets
    
    def get_test(self):
        return self.__testfeatures, self.__testtargets
    
    def get_binarizer(self):
        return self.__binarizer
    
    def get_classes(self):
        return self.get_binarizer().classes_

    def get_num_of_classes(self):
        return len(self.get_classes())

class TokenLevelClassificationData:
    
    __BACKUP_FILE_PREFIX = "TokenClassificationData_"
    __DEFAULT_TAG_TO_IGNORE = "O"
    __BACKUP_INTERVAL = 1000

    #BACKUP FIELDS
    __FEATURES_FIELD = "features"
    __TARGETS_FIELD = "targets"
    __CURRENTSENTENCEIDX_FIELD = "current_sentence_idx"
    __BINARIZER_FIELD = "binarizer"
    
    @staticmethod
    def load_from_backup(backup_dir:str):
        
        train_backup_path = None
        valid_backup_path = None
        test_backup_path = None
        
        if os.path.exists(backup_dir):
            train_backup_path = backup_dir + "/" + TokenLevelClassificationData.__BACKUP_FILE_PREFIX + "Train.pickle"
            valid_backup_path = backup_dir + "/" + TokenLevelClassificationData.__BACKUP_FILE_PREFIX + "Valid.pickle"
            test_backup_path = backup_dir + "/" + TokenLevelClassificationData.__BACKUP_FILE_PREFIX + "Test.pickle"
        
        train_backup_file = None
        valid_backup_file = None
        test_backup_file = None

        try:
            with open(train_backup_path, "rb") as infile:
                train_backup_file = pickle.load(infile)
            with open(valid_backup_path, "rb") as infile:
                valid_backup_file = pickle.load(infile)
            with open(test_backup_path, "rb") as infile:
                test_backup_file = pickle.load(infile)
            
            data = TokenLevelClassificationData(load_from_backup=True)
            
            data.__trainfeatures = train_backup_file[TokenLevelClassificationData.__FEATURES_FIELD]
            data.__traintargets = train_backup_file[TokenLevelClassificationData.__TARGETS_FIELD]
            data.__binarizer = train_backup_file[TokenLevelClassificationData.__BINARIZER_FIELD]

            data.__validfeatures = valid_backup_file[TokenLevelClassificationData.__FEATURES_FIELD]
            data.__validtargets = valid_backup_file[TokenLevelClassificationData.__TARGETS_FIELD]

            data.__testfeatures = test_backup_file[TokenLevelClassificationData.__FEATURES_FIELD]
            data.__testtargets = test_backup_file[TokenLevelClassificationData.__TARGETS_FIELD]

            return data
        except:
            raise Exception("Couldn't load the backup!")
    
    def __save_backup(self, backup_path:str, 
                      features:list, targets:list, current_sentence_idx:int, binarizer:LabelBinarizer, 
                      verbose:bool=False):
        
        if verbose:
            print("\nSaved backup {}\n".format(datetime.now().strftime("%d-%m-%y %H:%M:%S")))
        
        backup_data = {
            TokenLevelClassificationData.__FEATURES_FIELD: features,
            TokenLevelClassificationData.__TARGETS_FIELD: targets,
            TokenLevelClassificationData.__CURRENTSENTENCEIDX_FIELD: current_sentence_idx,
            TokenLevelClassificationData.__BINARIZER_FIELD: binarizer
            }
        
        try:
            if os.path.exists(backup_path):
                os.remove(backup_path)
            outfile = open(backup_path, "wb")
            pickle.dump(backup_data, outfile)
            outfile.close()
        except Exception as ex:
            print("Couldn't save the backup!\n" + str(ex))

    def __init__(self, 
                 train:pd.DataFrame=None, valid:pd.DataFrame=None, test:pd.DataFrame=None, 
                 tokenize_fn=None, 
                 sentence_idx_column_name:str="", word_column_name:str="", target_column_name:str="", 
                 max_len:int=128, verbose:bool=False, backup_dir:str="", load_from_backup:bool=False, 
                 tag_to_ignore:str=""):
        
        if load_from_backup:
            return
        
        if not isinstance(train, pd.DataFrame):
            raise Exception("No train data provided!")
        
        if not isinstance(valid, pd.DataFrame):
            raise Exception("No validation data provided!")
        
        if not isinstance(test, pd.DataFrame):
            raise Exception("No test data provided!")
        
        if tokenize_fn == None:
            raise Exception("No tokenize function provided!")
        
        if sentence_idx_column_name == "" or word_column_name == "" or target_column_name == "":
            raise Exception("No valid column names provided!")
        
        if max_len <= 2:
            raise Exception("Invalid max sequence length provided!")
        
        if tag_to_ignore == "":
            tag_to_ignore = TokenLevelClassificationData.__DEFAULT_TAG_TO_IGNORE
        
        self.__tokenize_fn = tokenize_fn
        self.__sentence_idx_column_name = sentence_idx_column_name
        self.__word_column_name = word_column_name
        self.__target_column_name = target_column_name
        self.__max_len = max_len
        self.__tag_to_ignore = tag_to_ignore

        self.__binarizer = LabelBinarizer()
        targets_to_fit = np.array( train[target_column_name].to_list() + [self.__tag_to_ignore] )
        self.__binarizer.fit(targets_to_fit)

        self.__backup_dir = None

        try:
            if backup_dir != "":
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                
                backup_dir = backup_dir.replace("\\", "/")
                if backup_dir[-1] == "/":
                    backup_dir = backup_dir[:-1]
                
                self.__backup_dir = backup_dir
        except:
            print("Couldn't create backup dir to the specified path.")
        
        self.__trainfeatures, self.__traintargets = self.__preprocess(train, verbose, "Train")
        self.__validfeatures, self.__validtargets = self.__preprocess(valid, verbose, "Valid")
        self.__testfeatures, self.__testtargets = self.__preprocess(test, verbose, "Test")

    def __preprocess(self, df:pd.DataFrame, verbose:bool, backup_name:str):
        
        if self.__sentence_idx_column_name not in df.columns or self.__word_column_name not in df.columns or self.__target_column_name not in df.columns:
            raise Exception("Invalid dataframe columns!")
        
        sentence_idx_column = df[self.__sentence_idx_column_name]
        word_column = df[self.__word_column_name]
        target_column = df[self.__target_column_name]
        target_column = self.__binarizer.transform(target_column.values)
        
        backup_path = None
        backup_file = None

        if self.__backup_dir != None and self.__backup_dir != "":
            backup_path = self.__backup_dir + "/" + TokenLevelClassificationData.__BACKUP_FILE_PREFIX + backup_name + ".pickle"

        if isinstance(backup_path, str):
            if os.path.exists(backup_path):
                try:
                    with open(backup_path, "rb") as infile:
                        backup_file = pickle.load(infile)
                except Exception as ex:
                    print("Couldn't load the backup!\n" + str(ex))
        
        if backup_file == None:
            features = []
            targets = []
            current_sentence_idx = sentence_idx_column[0]
        else:
            features = backup_file[TokenLevelClassificationData.__FEATURES_FIELD]
            targets = backup_file[TokenLevelClassificationData.__TARGETS_FIELD]
            current_sentence_idx = backup_file[TokenLevelClassificationData.__CURRENTSENTENCEIDX_FIELD]
        
        if verbose:
            to_iterate = tqdm(enumerate(word_column), total=len(word_column))
        else:
            to_iterate = enumerate(word_column)
        
        current_sentence_words = []
        current_sentence_classes = []

        for word_index, word in to_iterate:
            
            clean_word = re.sub(r'[\W_]', "", word)
            
            if clean_word == "":
                clean_word = word[0]
            
            sentence_idx = sentence_idx_column[word_index]
            target = target_column[word_index]
            
            if sentence_idx > current_sentence_idx:
                
                plaintext_current_sentence = [" ".join(current_sentence_words).strip()]
                preprocessed_current_sentence = self.__tokenize_fn(plaintext_current_sentence)[0]

                preprocessed_targets = []
                
                num_of_words = preprocessed_current_sentence.shape[0]
                if(num_of_words != len(current_sentence_classes)):
                    raise Exception("Inconsistency found, sentence index: " + str(current_sentence_idx))
                
                for preprocessedword_index, preprocessedword in enumerate(preprocessed_current_sentence):
                    num_of_tokens = preprocessedword.shape[0]
                    preprocessed_targets += [current_sentence_classes[preprocessedword_index]] * num_of_tokens
                
                preprocessed_targets = preprocessed_targets[:min(self.__max_len - 2, len(preprocessed_targets))]
                padlen = (self.__max_len - 2) - len(preprocessed_targets)
                padlen = max(padlen, 0)
                tag_to_ignore = np.array(self.__binarizer.transform([self.__tag_to_ignore])[0])
                preprocessed_targets = np.array( [tag_to_ignore] + preprocessed_targets + ([tag_to_ignore] * (padlen + 1)) )
                
                features.append(plaintext_current_sentence)
                targets.append(preprocessed_targets)

                current_sentence_words = []
                current_sentence_classes = []
                current_sentence_idx = sentence_idx
                
                if sentence_idx % TokenLevelClassificationData.__BACKUP_INTERVAL == 0:
                    self.__save_backup(backup_path, features, targets, current_sentence_idx, self.__binarizer, verbose)
            elif sentence_idx < current_sentence_idx:
                continue
            
            current_sentence_words.append(clean_word)
            current_sentence_classes.append(np.array(target))
        
        features = np.array(features)
        targets = np.array(targets)
        
        self.__save_backup(backup_path, features, targets, current_sentence_idx, self.__binarizer, verbose)

        return features, targets
    
    def get_train(self):
        return self.__trainfeatures, self.__traintargets
    
    def get_valid(self):
        return self.__validfeatures, self.__validtargets
    
    def get_test(self):
        return self.__testfeatures, self.__testtargets
    
    def get_binarizer(self):
        return self.__binarizer
    
    def get_classes(self):
        return self.__binarizer.classes_
    
    def get_num_of_classes(self):
        return len(self.get_classes())

class NLPClassificationModel(ABC):
    
    _ALIVE_FILE_NAME = "alive_data.pickle"
    
    _MODEL_TYPE_FIELD = "model_type"

    _SLCM_TYPE_NAME = "SLCM"
    _TLCM_TYPE_NAME = "TLCM"

    @staticmethod
    def load_model(path:str):
        
        path = path.replace("\\", "/")
        
        if path[-1] == "/":
            path = path[:-1]
        
        if not (os.path.exists(path) and os.path.isdir(path)):
            raise Exception("Given path is not an existing directory!")
        
        alive_data_path = path + "/" + NLPClassificationModel._ALIVE_FILE_NAME
        
        try:
            with open(alive_data_path, "rb") as infile:
                alive_data = pickle.load(infile)
                model_type = alive_data[NLPClassificationModel._MODEL_TYPE_FIELD]
        except:
            model_type = "None"
        
        if model_type == NLPClassificationModel._SLCM_TYPE_NAME:
            model = SentenceLevelClassificationModel("")
        elif model_type == NLPClassificationModel._TLCM_TYPE_NAME:
            model = TokenLevelClassificationModel("")
        else:
            raise Exception("Couldn't detect model type!")
        
        try:
            model._model = tf.keras.models.load_model(path)
        except:
            raise Exception("Couldn't load the model!")
        
        try:
            model._initialize_from_alive_file(alive_data_path)
        except:
            raise Exception("Couldn't load the AlIve file!")
        
        return model
    
    def save(self, path:str, overwrite:bool=False):
        
        if not isinstance(self._model, tf.keras.Model):
            raise Exception("The model is not built! Can't save it!")
        
        path = path.replace("\\", "/")

        if path[-1] == "/":
            path = path[:-1]
        
        invalid_chars = [",", ".", ";", ":", "*", "<", ">"]

        for invalid_char in invalid_chars:
            if invalid_char in path:
                raise Exception("Given path is not a valid directory!")
        
        if os.path.exists(path):
            if os.path.isdir(path):
                if not overwrite:
                    raise Exception("Directory already existing!")
            else:
                raise Exception("Can't save the model to an existing file!")
        else:        
            try:
                os.makedirs(path)
            except:
                raise Exception("Couldn't create the directory with the given path!")
        
        self._model.save(path, overwrite)
        
        alive_data_path = path + "/" + NLPClassificationModel._ALIVE_FILE_NAME
        alive_data = self._get_alive_data()
        
        try:
            if os.path.exists(alive_data_path):
                os.remove(alive_data_path)
            outfile = open(alive_data_path, "wb")
            pickle.dump(alive_data, outfile)
            outfile.close()
        except:
            raise Exception("Couldn't save the AlIve file!")
    
    @abstractmethod
    def _initialize_from_alive_file(self, alive_data_path:str):
        pass
    
    @abstractmethod
    def _get_alive_data(self):
        pass
    
    def __init__(self, name:str="", finetunable:bool=False):
        self._model = None
        self._name = name
        self._binarizer = None
        self._finetunable = finetunable
        self._already_trained = False
        self._histories = dict()
    
    def get_name(self):
        return self._name
    
    def set_name(self, new_name:str):

        if new_name == "":
            return

        self._name = new_name

        if not isinstance(self._model, tf.keras.Model):
            return
        
        self._model._name = self._name
    
    def get_histories(self):
        return self._histories
    
    def get_most_recent_history(self):

        if len(self._histories.keys()) <= 0:
            return None, None
        
        dates = list(self._histories.keys())
        dates.sort()
        most_recent_date = dates[-1]

        return self._histories[most_recent_date], most_recent_date
    
    @abstractmethod
    def summary(self):
        pass
    
    @abstractmethod
    def evaluate(self, testfeatures, testtargets):
        pass
    
    @abstractmethod
    def predict(self, example):
        pass
    
    def save_train_history_graph(self, history_dict:dict, date:datetime, directory:str, 
                                 metrics_to_plot:list=None):
        
        if metrics_to_plot == None:
            metrics_to_plot = list(history_dict.keys())
        
        for metric_to_plot in metrics_to_plot:
            if metric_to_plot not in history_dict.keys():
                raise Exception("Invalid metric to plot!")
        
        epochs = range(1, len(history_dict[metrics_to_plot[0]]) + 1)
        fig = plt.figure(figsize=(10, 8))
        fig.tight_layout()

        cmap = plt.cm.get_cmap("hsv", len(metrics_to_plot) + 1)
        
        for i, metric_to_plot in enumerate(metrics_to_plot):
            plt.plot(epochs, history_dict[metric_to_plot], color=cmap(i), label=metric_to_plot)
        
        shown_model_name = self._name if self._name != "" else "your model"

        plt.title("Training History for " + shown_model_name + " TRAINING COMPLETED ON: " + date.strftime("%d-%m-%y %H:%M:%S"))
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel("Metrics")
        plt.legend()
        
        directory = directory.replace("\\", "/")

        if directory[-1] == "/":
            directory = directory[:-1]
        
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except:
                raise Exception("Invalid path given!")
        else:
            if os.path.isfile(directory):
                raise Exception("Invalid path given!")
        
        save_path = directory + "/" + self._name + date.strftime("%d-%m-%y_%H:%M:%S") + ".png"

        if os.path.exists(save_path):
            os.remove(save_path)

        plt.savefig(save_path)

class SentenceLevelClassificationModel(NLPClassificationModel):

    #ALIVE FILE FIELDS
    __NAME_FIELD = "name"
    __BINARIZER_FIELD = "binarizer"
    __FINETUNABLE_FIELD = "finetunable"
    __ALREADY_TRAINED_FIELD = "already_trained"
    __HISTORIES_FIELD = "histories"

    def _initialize_from_alive_file(self, alive_data_path:str):

        with open(alive_data_path, "rb") as infile:
            alive_data = pickle.load(infile)
            
            self._name = alive_data[SentenceLevelClassificationModel.__NAME_FIELD]
            self._binarizer = alive_data[SentenceLevelClassificationModel.__BINARIZER_FIELD]
            self._finetunable = alive_data[SentenceLevelClassificationModel.__FINETUNABLE_FIELD]
            self._already_trained = alive_data[SentenceLevelClassificationModel.__ALREADY_TRAINED_FIELD]
            self._histories = alive_data[SentenceLevelClassificationModel.__HISTORIES_FIELD]

    def _get_alive_data(self):

        alive_data = {
            NLPClassificationModel._MODEL_TYPE_FIELD : NLPClassificationModel._SLCM_TYPE_NAME,
            SentenceLevelClassificationModel.__NAME_FIELD : self._name,
            SentenceLevelClassificationModel.__BINARIZER_FIELD: self._binarizer,
            SentenceLevelClassificationModel.__FINETUNABLE_FIELD: self._finetunable,
            SentenceLevelClassificationModel.__ALREADY_TRAINED_FIELD : self._already_trained,
            SentenceLevelClassificationModel.__HISTORIES_FIELD : self._histories
        }

        return alive_data

    def __init__(self, name:str="", finetunable:bool=False):
        super().__init__(name, finetunable)
    
    def build(self, encoder_model_link:str, output_shape:int, 
              preprocess_model_link:str=None, encoder_trainable:bool=False, encoder_output_key:str=None, 
              dropout_rate:float=0.1, final_output_activation=None, 
              optimizer_lr:float=1e-5, additional_metrics:list=None, run_eagerly:bool=True):
        
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

        if preprocess_model_link != None:
            preprocessing = hub.KerasLayer(preprocess_model_link, name='preprocessing')
            encoder_inputs = preprocessing(text_input)
        else:
            encoder_inputs = text_input
        
        encoder = hub.KerasLayer(encoder_model_link, trainable=encoder_trainable, name='encoder')
        encoder_output = encoder(encoder_inputs)
        
        if encoder_output_key == None:
            net = encoder_output
        else:
            try:
                net = encoder_output[encoder_output_key]
            except:
                net = encoder_output
        
        net = tf.keras.layers.Dropout(dropout_rate)(net)
        net = tf.keras.layers.Dense(output_shape, activation=final_output_activation, name='classifier')(net)

        if self._name == "":
            self._model = tf.keras.Model(text_input, net)
            self._name = self._model._name
        else:
            self._model = tf.keras.Model(text_input, net, name=str(self._name))
        
        if not isinstance(additional_metrics, list):
            additional_metrics = []
        
        optimizer = tf.keras.optimizers.Adam(optimizer_lr)

        if output_shape > 2:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metrics = [tf.metrics.CategoricalAccuracy()] + additional_metrics
        else:
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = [tf.metrics.BinaryAccuracy()] + additional_metrics
        
        self._model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            run_eagerly=run_eagerly)
    
    def train(self, data:SentenceLevelClassificationData, epochs:int, batch_size:int=32, 
              checkpoint_path:str="", desired_callbacks:list=None):
        
        if not isinstance(self._model, tf.keras.Model):
            raise Exception("Model is not built!")
        
        if (not self._finetunable) and self._already_trained:
            raise Exception("This model can't be trained again!")
        
        if self._binarizer == None:
            self._binarizer = data.get_binarizer()
        else:
            for new_class in data.get_binarizer().classes_:
                if new_class not in self._binarizer.classes_:
                    raise Exception("Invalid classes found in the dataset for this train session!")

        model = self._model

        trainfeatures, traintargets = data.get_train()
        validfeatures, validtargets = data.get_valid()

        if desired_callbacks == None:
            desired_callbacks = []

        if checkpoint_path != "":
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             verbose=1)
            desired_callbacks.append(cp_callback)
            
            if os.path.exists(checkpoint_path):

                print("\nTrying to load from specified checkpoint path.\n")

                try:
                    model.load_weights(checkpoint_path)
                except:
                    print("\nCould not load from checkpoint...\n")
        
        history = model.fit(x=trainfeatures, y=traintargets,
                            validation_data=(validfeatures, validtargets),
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=desired_callbacks)
        
        ending_time = datetime.now()
        
        self._already_trained = True
        self._histories[ending_time] = history.history

        return history, ending_time
    
    def summary(self):

        if not isinstance(self._model, tf.keras.Model):
            raise Exception("Model not built!")
        
        self._model.summary()
    
    def evaluate(self, testfeatures, testtargets):

        if not isinstance(self._model, tf.keras.Model):
            raise Exception("Model not built!")
        
        loss, accuracy = self._model.evaluate(testfeatures, testtargets)
        return loss, accuracy
    
    def predict(self, example):

        if not isinstance(self._model, tf.keras.Model):
            raise Exception("Model not built!")
        
        if self._binarizer == None:
            raise Exception("Binarizer doesn't exist!")
        
        result = self._model(tf.constant([example]))

        max_prob = 0

        if result[0].shape[0] > 1:

            result = tf.nn.softmax(result)
            
            for prob in result[0]:

                prob = float(prob)

                if prob > max_prob:
                    max_prob = prob

        elif result[0].shape[0] == 1:
            prob = float(tf.nn.sigmoid(result))

            if prob >= 0.5:
                max_prob = prob
            else:
                max_prob = 1 - prob
        
        prediction = self._binarizer.inverse_transform(result.numpy())[0]

        return prediction, max_prob

class TokenLevelClassificationModel(NLPClassificationModel):

    #ALIVE FILE FIELDS
    __NAME_FIELD = "name"
    __BINARIZER_FIELD = "binarizer"
    __FINETUNABLE_FIELD = "finetunable"
    __ALREADY_TRAINED_FIELD = "already_trained"
    __HISTORIES_FIELD = "histories"

    def _initialize_from_alive_file(self, alive_data_path:str):

        with open(alive_data_path, "rb") as infile:
            alive_data = pickle.load(infile)
            
            self._name = alive_data[TokenLevelClassificationModel.__NAME_FIELD]
            self._binarizer = alive_data[TokenLevelClassificationModel.__BINARIZER_FIELD]
            self._finetunable = alive_data[TokenLevelClassificationModel.__FINETUNABLE_FIELD]
            self._already_trained = alive_data[TokenLevelClassificationModel.__ALREADY_TRAINED_FIELD]
            self._histories = alive_data[TokenLevelClassificationModel.__HISTORIES_FIELD]

    def _get_alive_data(self):

        alive_data = {
            NLPClassificationModel._MODEL_TYPE_FIELD : NLPClassificationModel._TLCM_TYPE_NAME,
            TokenLevelClassificationModel.__NAME_FIELD : self._name,
            TokenLevelClassificationModel.__BINARIZER_FIELD: self._binarizer,
            TokenLevelClassificationModel.__FINETUNABLE_FIELD: self._finetunable,
            TokenLevelClassificationModel.__ALREADY_TRAINED_FIELD : self._already_trained,
            TokenLevelClassificationModel.__HISTORIES_FIELD : self._histories
        }

        return alive_data

    def __init__(self, name:str="", finetunable:bool=False):
        super().__init__(name, finetunable)
    
    def build(self, preprocess_model_link:str, encoder_model_link:str, output_shape:int, 
              encoder_trainable:bool=False, encoder_output_key:str="sequence_output", 
              dropout_rate:float=0.3, final_output_activation=None, 
              optimizer_lr:float=1e-5, additional_metrics:list=None, run_eagerly:bool=True):
        
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        
        preprocessor = tf.keras.models.load_model(hub.resolve(preprocess_model_link))

        tokenizer_layer = preprocessor.get_layer("bert_tokenizer")
        bert_pack_inputs_layer = preprocessor.get_layer("bert_pack_inputs")

        encoder_inputs = bert_pack_inputs_layer(tokenizer_layer(text_input))

        encoder = hub.KerasLayer(encoder_model_link, trainable=encoder_trainable, name='encoder')
        
        if encoder_output_key == None:
            encoder_output = encoder(encoder_inputs)
        else:
            try:
                encoder_output = encoder(encoder_inputs)[encoder_output_key]
            except:
                encoder_output = encoder(encoder_inputs)
        
        embedding = tf.keras.layers.Dropout(dropout_rate)(encoder_output)
        final_output = tf.keras.layers.Dense(output_shape, activation=final_output_activation)(embedding)

        if self._name == "":
            self._model = tf.keras.Model(inputs = [text_input], outputs = [final_output])
            self._name = self._model._name
        else:
            self._model = tf.keras.Model(inputs = [text_input], outputs = [final_output], name=str(self._name))
        
        if not isinstance(additional_metrics, list):
            additional_metrics = []
        
        optimizer = tf.keras.optimizers.Adam(optimizer_lr)
        
        if output_shape > 2:
            loss = "categorical_crossentropy"
            metrics = ["accuracy"] + additional_metrics
        else:
            loss = "binary_crossentropy"
            metrics = ["accuracy"] + additional_metrics
        
        self._model.compile(optimizer=optimizer, 
                            loss=loss, 
                            metrics=metrics, 
                            run_eagerly=run_eagerly)
    
    def train(self, data:TokenLevelClassificationData, epochs:int, batch_size:int=32, 
              checkpoint_path:str="", desired_callbacks:list=None):
        
        if not isinstance(self._model, tf.keras.Model):
            raise Exception("Model is not built!")
        
        if (not self._finetunable) and (self._already_trained):
            raise Exception("This model can't be trained again!")
        
        if self._binarizer == None:
            self._binarizer = data.get_binarizer()
        else:
            for new_class in data.get_binarizer().classes_:
                if new_class not in self._binarizer.classes_:
                    raise Exception("Invalid classes found in the dataset for this train session!")
        
        model = self._model

        trainfeatures, traintargets = data.get_train()
        validfeatures, validtargets = data.get_valid()

        if desired_callbacks == None:
            desired_callbacks = []
        
        if checkpoint_path != "":
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             verbose=1)
            desired_callbacks.append(cp_callback)
            
            if os.path.exists(checkpoint_path):

                print("\nTrying to load from specified checkpoint path.\n")

                try:
                    model.load_weights(checkpoint_path)
                except:
                    print("\nCould not load from checkpoint...\n")
        
        history = model.fit(x=trainfeatures, y=traintargets,
                            validation_data=(validfeatures, validtargets),
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=desired_callbacks)
        
        ending_time = datetime.now()
        
        self._already_trained = True
        self._histories[ending_time] = history.history

        return history, ending_time
    
    def summary(self):

        if not isinstance(self._model, tf.keras.Model):
            raise Exception("Model not built!")
        
        self._model.summary()
    
    def evaluate(self, testfeatures, testtargets):

        if not isinstance(self._model, tf.keras.Model):
            raise Exception("Model not built!")

        loss, accuracy = self._model.evaluate(testfeatures, testtargets)
        return loss, accuracy
    
    def predict(self, example):

        example = self.preprocess_example(example)
        words_list = re.split(r"\s+", example)

        if not isinstance(self._model, tf.keras.Model):
            raise Exception("Model not built!")
        
        if self._binarizer == None:
            raise Exception("Binarizer doesn't exist!")
        
        sentenceprediction = np.array( self._model(tf.constant([example]))[0] )
        
        if sentenceprediction[0].shape[0] > 1:
            sentenceprediction = tf.nn.softmax(sentenceprediction)
        
        sentenceprediction = self._binarizer.inverse_transform(sentenceprediction.numpy())[1:]

        decoded_prediction = []

        tokenizedexample = self.tokenize(example)

        for word in tokenizedexample:
            num_of_tokens = word.shape[0]
            word_class = sentenceprediction[0]
            decoded_prediction.append(word_class)
            sentenceprediction = sentenceprediction[num_of_tokens:]
        
        result = np.array(decoded_prediction)

        entities = []

        for word, index in words_list:
            entity_prediction = result[index]

            if (entity_prediction != "O"):
                entity_tuple = (word, entity_prediction)
                entities.append(entity_tuple)
        
        print(entities)

        return result, entities
    
    def preprocess_example(self, example):
        
        clean_example = re.sub(r'[\W_]', " ", example)

        return clean_example

    def tokenize(self, sentence):

        layer_name = 'bert_tokenizer'
        tokenizer_model = tf.keras.Model(inputs=self._model.input, 
                                        outputs=self._model.get_layer(layer_name).output)
        tokenized_sentence = tokenizer_model(tf.constant([sentence]))[0]

        return tokenized_sentence

def get_handle_preprocess_link(model_name:str):

    map_name_to_handle = {
        'bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3',
        'bert_en_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3',
        'bert_multi_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
        'small_bert/bert_en_uncased_L-2_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-2_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-2_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-2_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-4_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-4_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-4_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-4_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-6_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-6_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-6_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-6_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-8_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-8_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-8_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-8_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-10_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-10_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-10_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-10_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1',
        'small_bert/bert_en_uncased_L-12_H-128_A-2':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1',
        'small_bert/bert_en_uncased_L-12_H-256_A-4':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1',
        'small_bert/bert_en_uncased_L-12_H-512_A-8':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1',
        'small_bert/bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1',
        'albert_en_base':
            'https://tfhub.dev/tensorflow/albert_en_base/2',
        'electra_small':
            'https://tfhub.dev/google/electra_small/2',
        'electra_base':
            'https://tfhub.dev/google/electra_base/2',
        'experts_pubmed':
            'https://tfhub.dev/google/experts/bert/pubmed/2',
        'experts_wiki_books':
            'https://tfhub.dev/google/experts/bert/wiki_books/2',
        'talking-heads_base':
            'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1',
        'universal-sentence-encoder-cmlm/en-base':
            'https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1',
        'universal-sentence-encoder-cmlm/multilingual-base':
            'https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1',
        'universal-sentence-encoder-xling/en-es':
            'https://tfhub.dev/google/universal-sentence-encoder-xling/en-es/1',
        'universal-sentence-encoder-xling/en-de':
            'https://tfhub.dev/google/universal-sentence-encoder-xling/en-de/1',
        'universal-sentence-encoder-xling-many':
            'https://tfhub.dev/google/universal-sentence-encoder-xling-many/1',
        'universal-sentence-encoder':
            'https://tfhub.dev/google/universal-sentence-encoder/4',
        'universal-sentence-encoder-lite':
            'https://tfhub.dev/google/universal-sentence-encoder-lite/2',
        'universal-sentence-encoder-large':
            'https://tfhub.dev/google/universal-sentence-encoder-large/5',
        'universal-sentence-encoder-multilingual':
            'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3',
        'universal-sentence-encoder-multilingual-large':
            'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
    }

    map_model_to_preprocess = {
        'bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'bert_en_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/2',
        'small_bert/bert_en_uncased_L-2_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-2_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-2_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-2_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-4_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-4_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-4_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-4_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-6_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-6_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-6_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-6_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-8_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-8_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-8_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-8_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-10_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-10_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-10_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-10_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-12_H-128_A-2':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-12_H-256_A-4':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-12_H-512_A-8':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'small_bert/bert_en_uncased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'bert_multi_cased_L-12_H-768_A-12':
            'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2',
        'albert_en_base':
            'https://tfhub.dev/tensorflow/albert_en_preprocess/2',
        'electra_small':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'electra_base':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'experts_pubmed':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'experts_wiki_books':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'talking-heads_base':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2',
        'universal-sentence-encoder-cmlm/en-base':
            'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        'universal-sentence-encoder-cmlm/multilingual-base':
            'https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2',
        'universal-sentence-encoder-xling/en-es':
            None,
        'universal-sentence-encoder-xling/en-de':
            None,
        'universal-sentence-encoder-xling-many':
            None,
        'universal-sentence-encoder':
            None,
        'universal-sentence-encoder-lite':
            None,
        'universal-sentence-encoder-large':
            None,
        'universal-sentence-encoder-multilingual':
            None,
        'universal-sentence-encoder-multilingual-large':
            None
    }

    tfhub_handle_handle = map_name_to_handle[model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[model_name]
    
    return tfhub_handle_handle, tfhub_handle_preprocess

def get_available_base_models():
    
    available_models = [
        'bert_en_uncased_L-12_H-768_A-12',
        'bert_en_cased_L-12_H-768_A-12',
        'bert_multi_cased_L-12_H-768_A-12',
        'small_bert/bert_en_uncased_L-2_H-128_A-2',
        'small_bert/bert_en_uncased_L-2_H-256_A-4',
        'small_bert/bert_en_uncased_L-2_H-512_A-8',
        'small_bert/bert_en_uncased_L-2_H-768_A-12',
        'small_bert/bert_en_uncased_L-4_H-128_A-2',
        'small_bert/bert_en_uncased_L-4_H-256_A-4',
        'small_bert/bert_en_uncased_L-4_H-512_A-8',
        'small_bert/bert_en_uncased_L-4_H-768_A-12',
        'small_bert/bert_en_uncased_L-6_H-128_A-2',
        'small_bert/bert_en_uncased_L-6_H-256_A-4',
        'small_bert/bert_en_uncased_L-6_H-512_A-8',
        'small_bert/bert_en_uncased_L-6_H-768_A-12',
        'small_bert/bert_en_uncased_L-8_H-128_A-2',
        'small_bert/bert_en_uncased_L-8_H-256_A-4',
        'small_bert/bert_en_uncased_L-8_H-512_A-8',
        'small_bert/bert_en_uncased_L-8_H-768_A-12',
        'small_bert/bert_en_uncased_L-10_H-128_A-2',
        'small_bert/bert_en_uncased_L-10_H-256_A-4',
        'small_bert/bert_en_uncased_L-10_H-512_A-8',
        'small_bert/bert_en_uncased_L-10_H-768_A-12',
        'small_bert/bert_en_uncased_L-12_H-128_A-2',
        'small_bert/bert_en_uncased_L-12_H-256_A-4',
        'small_bert/bert_en_uncased_L-12_H-512_A-8',
        'small_bert/bert_en_uncased_L-12_H-768_A-12',
        'albert_en_base',
        'electra_small',
        'electra_base',
        'experts_pubmed',
        'experts_wiki_books',
        'talking-heads_base',
        'universal-sentence-encoder-cmlm/en-base',
        'universal-sentence-encoder-cmlm/multilingual-base',
        'universal-sentence-encoder-xling/en-es',
        'universal-sentence-encoder-xling/en-de',
        'universal-sentence-encoder-xling-many',
        'universal-sentence-encoder',
        'universal-sentence-encoder-lite',
        'universal-sentence-encoder-large',
        'universal-sentence-encoder-multilingual',
        'universal-sentence-encoder-multilingual-large'
    ]

    return available_models

class EntitiesHierarchy:
    
    ITALIAN_LANGUAGE_FIELD_NAME = "ita"
    ENGLISH_LANGUAGE_FIELD_NAME = "en"
    GERMAN_LANGUAGE_FIELD_NAME = "de"
    
    SUPPORTED_LANGUAGES = [ITALIAN_LANGUAGE_FIELD_NAME, 
                           ENGLISH_LANGUAGE_FIELD_NAME, 
                           GERMAN_LANGUAGE_FIELD_NAME]
    
    ROOT_TAG = "Entity"
    ROOT_ID = "entity"
    ROOT_ITALIAN_SYNSET = {"entità", "ente"}
    ROOT_ENGLISH_SYNSET = {"entity"}
    
    def __init__(self):
        
        entity_synset = {
            EntitiesHierarchy.ITALIAN_LANGUAGE_FIELD_NAME : EntitiesHierarchy.ROOT_ITALIAN_SYNSET,
            EntitiesHierarchy.ENGLISH_LANGUAGE_FIELD_NAME : EntitiesHierarchy.ROOT_ENGLISH_SYNSET
        }
        
        self.__tree = Tree()
        self.__root = self.__tree.create_node(EntitiesHierarchy.ROOT_TAG, 
                                              EntitiesHierarchy.ROOT_ID, 
                                              data=entity_synset)
    
    def add_node(self, new_node_tag:str, new_node_id:str, parent_id:str=None, sysnset:dict=None):
        
        if self.__tree.get_node(new_node_id) != None:
            raise Exception("A node with this id already exists!")
        
        if parent_id == None:
            parent_id = EntitiesHierarchy.ROOT_ID
        
        if sysnset == None:
            sysnset = dict()
            
            for language in EntitiesHierarchy.SUPPORTED_LANGUAGES:
                sysnset[language] = set()
        
        if self.__tree.get_node(parent_id) == None:
            raise Exception("Parent node with this id doesn't exist!")
        
        self.__tree.create_node(new_node_tag, new_node_id, parent_id, sysnset)
    
    def remove_node(self, node_id:str):

        if node_id == EntitiesHierarchy.ROOT_ID:
            raise Exception("Can't remove the root!")
        
        if self.__tree.get_node(node_id) == None:
            raise Exception("A node with this id doesn't exist!")
        
        self.__tree.remove_node(node_id)
    
    def move_node(self, node_id:str, new_parent_id:str):
        
        if self.__tree.get_node(node_id) == None:
            raise Exception("The node you want to move doesn't exist!")
        
        if self.__tree.get_node(new_parent_id) == None:
            raise Exception("Parent node with this id doesn't exist!")
        
        self.__tree.move_node(node_id, new_parent_id)
    
    def add_terms_to_synset(self, node_id:str, language:str, terms:set):
        
        desired_node = self.__tree.get_node(node_id)
        
        if desired_node == None:
            raise Exception("A node with this id doesn't exist!")
        
        synset = desired_node.data
        
        if language not in synset:
            synset[language] = set()
        
        for term in terms:
            synset[language].add(term)
        
        desired_node.data = synset
    
    def remove_terms_to_synset(self, node_id:str, language:str, terms:set):
        
        desired_node = self.__tree.get_node(node_id)
        
        if desired_node == None:
            raise Exception("A node with this id doesn't exist!")
        
        synset = desired_node.data
        
        if language not in synset:
            raise Exception("The synset for this node doesn't have this language!")
        
        for term in terms:
            if term in synset[language]:
                synset[language].remove(term)
        
        desired_node.data = synset
    
    def reset_synset(self, node_id:str, languages:list=None):
        
        desired_node = self.__tree.get_node(node_id)
        
        if desired_node == None:
            raise Exception("A node with this id doesn't exist!")
        
        synset = desired_node.data
        
        if languages == None:
            languages = synset.keys()
        
        for language in languages:
            if language in synset.keys():
                del( synset[language] )
                synset[language] = set()
        
        desired_node.data = synset
