import os
from abc import ABC, abstractmethod
from queue import Queue
import pickle

import pandas as pd

from ModelsAndDatasets import *

#CONSTANTS
TEXT_COLUMN_NAME = "text"
SENTENCE_IDX_COLUMN_NAME = "sentence_idx"
WORD_COLUMN_NAME = "word"
EXAMPLE_CATEGORY_COLUMN_NAME = "example_category"

EXAMPLE_TRAIN_CATEGORY = "train"
EXAMPLE_VALIDATION_CATEGORY = "valid"
EXAMPLE_TEST_CATEGORY = "test"

NULL_TARGET_VALUE = ""

SLC_DATASET_TYPE_NAME = "SLCDataset"
TLC_DATASET_TYPE_NAME = "TLCDataset"
SLC_MODEL_TYPE_NAME = "SLCModel"
TLC_MODEL_TYPE_NAME = "TLCModel"

MAX_TRAIN_QUEUE_SIZE = 3

ROOT_FOLDER = "AlIve"
METADATA_FILE_NAME = "Metadata.pickle"
USERS_DATA_FOLDER_NAME = "UsersData"
AMBIENTS_FOLDER_NAME = "Ambients"
DATASETS_FOLDER_NAME = "Datasets"
MODELS_FOLDER_NAME = "Models"

METADATA_PATH = ROOT_FOLDER + "/" + METADATA_FILE_NAME
USERS_DATA_FOLDER = ROOT_FOLDER + "/" + USERS_DATA_FOLDER_NAME + "/"
#END OF CONSTANTS

class AliveUser:

    def __init__(self, username:str, password:str):
        self.username = username
        self.password = password
        self.ambients = dict()
    
    def create_ambient(self, new_ambient_name:str):
        
        if new_ambient_name in self.ambients.keys():
            raise Exception("An ambient with this name already exists!")

        self.ambients[new_ambient_name] = AliveAmbient(self, new_ambient_name)
    
    def rename_ambient(self, old_ambient_name:str, new_ambient_name:str):
        
        if old_ambient_name not in self.ambients.keys():
            raise Exception("An ambient with this name doesn't exist!")
        
        ambient_to_rename = self.ambients[old_ambient_name]
        ambient_to_rename.set_name(new_ambient_name)

        self.ambients[new_ambient_name] = ambient_to_rename
        del self.ambients[old_ambient_name]

    def delete_ambient(self, ambient_name:str):
        if ambient_name in self.ambients.keys():
            del self.ambients[ambient_name]

class AliveAmbient:

    def __init__(self, creator:AliveUser, name:str):
        self.__creator = creator
        self.__name = name

        self.__datasets = dict()
        self.__models = dict()
        
        self.__train_queue = Queue(maxsize=MAX_TRAIN_QUEUE_SIZE)
    
    def get_creator(self):
        return self.__creator

    def get_name(self):
        return self._name

    def set_name(self, name:str):
        self._name = name

    def add_dataset(self, dataset_name:str, dataset_type:str):

        if dataset_name in self.__datasets.keys():
            raise Exception("A dataset with this name already exists in the ambient!")
        
        dataframe_dict = dict()
        
        if dataset_type == SLC_DATASET_TYPE_NAME:
            dataframe_dict[TEXT_COLUMN_NAME] = []
        elif dataset_type == TLC_DATASET_TYPE_NAME:
            dataframe_dict[SENTENCE_IDX_COLUMN_NAME] = []
            dataframe_dict[WORD_COLUMN_NAME] = []
        else:
            raise Exception("Invalid dataset type!")
        
        dataset = pd.DataFrame(dataframe_dict)

        dataset_dir = ROOT_FOLDER + "/" + self.__creator + "/" + self.__name + "/" + DATASETS_FOLDER_NAME + "/"
        dataset_path = dataset_dir + "/" + dataset_name + ".pickle"

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        try:
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
            
            dataset.to_pickle(dataset_path)
        except Exception as ex:
            raise Exception("Couldn't create the dataset!")
        
        dataset_metadata = DatasetMetadata(dataset_name, dataset_path, dataset_type)

        self.__datasets[dataset_name] = dataset_metadata

    def import_examples_to_dataset(self, dataset_name:str, imported_dataset, category:str, 
                                   text_column_name:str=TEXT_COLUMN_NAME, 
                                   sentence_idx_column_name:str=SENTENCE_IDX_COLUMN_NAME, word_column_name:str=WORD_COLUMN_NAME):

        if(dataset_name not in self.__datasets.keys()):
            raise Exception("The dataset called : '" + dataset_name + "' doesn't exist!")
        
        existing_dataset_path = self.__datasets[dataset_name].get_path()
        existing_dataset_type = self.__datasets[dataset_name].get_type()

        try:
            with open(existing_dataset_path, "rb") as infile:
                existing_dataset = pickle.load(infile)
        except:
            raise Exception("Couldn't load the '" + dataset_name + "' dataset!")
        
        if not isinstance(imported_dataset, pd.DataFrame):
            try:
                imported_dataset = pd.read_csv(str(imported_dataset))
            except:
                raise Exception("Cannot import examples!")
        
        needed_fields = []

        if text_column_name in imported_dataset.columns:
            imported_dataset.rename(columns={text_column_name: TEXT_COLUMN_NAME})
        
        if sentence_idx_column_name in imported_dataset.columns:
            imported_dataset.rename(columns={sentence_idx_column_name: SENTENCE_IDX_COLUMN_NAME})
        
        if word_column_name in imported_dataset.columns:
            imported_dataset.rename(columns={word_column_name: WORD_COLUMN_NAME})
        
        if existing_dataset_type == SLC_DATASET_TYPE_NAME:
            needed_fields += [TEXT_COLUMN_NAME]
        elif existing_dataset_type == TLC_DATASET_TYPE_NAME:
            needed_fields += [SENTENCE_IDX_COLUMN_NAME, WORD_COLUMN_NAME]
        
        for needed_field in needed_fields:
            if needed_field not in imported_dataset.columns:
                raise Exception("Trying to import from an invalid dataframe!")
        
        for column in imported_dataset.columns:
            if column not in existing_dataset:
                existing_dataset[column] = NULL_TARGET_VALUE
        
        for i, new_row in imported_dataset.iterrows():

            for column in existing_dataset:
                if column not in imported_dataset:
                    new_row[column] = NULL_TARGET_VALUE

            existing_dataset.append(new_row, ignore_index=True)
        
        try:
            if os.path.exists(existing_dataset_path):
                os.remove(existing_dataset_path)
            
            existing_dataset_path.to_pickle(existing_dataset_path)
        except Exception as ex:
            raise Exception("Couldn't save the updated dataset!")

    def create_slcmodel(self, 
                        model_name:str, finetunable:bool, 
                        encoder_model_link:str, dataset:pd.DataFrame, target_column_name, 
                        preprocess_model_link:str="", encoder_trainable:bool=False, encoder_output_key:str="", 
                        dropout_rate:float=0.1, final_output_activation=None, 
                        optimizer_lr:float=1e-5, additional_metrics:list=None, run_eagerly:bool=True):
        
        if model_name in self.__models.keys():
            raise Exception("A model with this name already belongs to the ambient!")
        
        model_dir = ROOT_FOLDER + "/" + self.__creator + "/" + self.__name + "/" + MODELS_FOLDER_NAME + "/"
        model_path = model_dir + "/" + model_name + "/"

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        train = dataset[dataset[EXAMPLE_CATEGORY_COLUMN_NAME] == EXAMPLE_TRAIN_CATEGORY]
        valid = dataset[dataset[EXAMPLE_CATEGORY_COLUMN_NAME] == EXAMPLE_VALIDATION_CATEGORY]
        test = dataset[dataset[EXAMPLE_CATEGORY_COLUMN_NAME] == EXAMPLE_TEST_CATEGORY]

        ic_data = SentenceLevelClassificationData(train, valid, test, TEXT_COLUMN_NAME, target_column_name)
        num_of_classes = ic_data.get_num_of_classes()
        
        new_slc_model = SentenceLevelClassificationModel(model_name, finetunable)
        new_slc_model.build(encoder_model_link, num_of_classes, preprocess_model_link, 
                            encoder_trainable, encoder_output_key, 
                            dropout_rate, final_output_activation, 
                            optimizer_lr, additional_metrics, run_eagerly)
        
        new_slc_model.save(model_path)

        model_metadata = ModelMetadata(model_name, model_path, SLC_MODEL_TYPE_NAME)

        self.__models[model_name] = model_metadata

    def add_model_to_train_queue(self, model):
        self.__train_queue.put(model)

    def start_training_models_in_queue(self):
        self.__train_queue.get()

    def train_model(self, model):
        self.__train_queue.get()

class Metadata(ABC):

    def __init__(self, name, path, type):
        self._name = name
        self._path = path
        self._type = type
    
    def set_name(self, name):
        self._name = name

    def get_name(self):
        return self._name
    
    def set_path(self, path):
        self._path = path

    def get_path(self):
        return self._path
    
    def get_type(self):
        return self._type
    
    @abstractmethod
    def load(self):
        pass

class DatasetMetadata(Metadata):
        
    def __init__(self, name, path, type):
        super().__init__(name, path, type)
    
    def load(self):
        
        needed_fields = []

        if self._type == SLC_DATASET_TYPE_NAME:
            needed_fields += [TEXT_COLUMN_NAME]
        elif self._type == TLC_DATASET_TYPE_NAME:
            needed_fields += [SENTENCE_IDX_COLUMN_NAME, WORD_COLUMN_NAME]

        try:
            with open(self._path, "rb") as infile:
                dataframe = pickle.load(infile)
                
                if not isinstance(dataframe, pd.DataFrame):
                    raise Exception("Couldn't load dataset!")
                
                for needed_field in needed_fields:
                    if needed_field not in dataframe.columns:
                        raise Exception("Malformed dataset!")
                    
                return dataframe
        except Exception as ex:
            raise ex

class ModelMetadata(Metadata):
        
    def __init__(self, name, path, type):
        super().__init__(name, path, type)
    
    def load(self):
        return NLPClassificationModel.load_model(self._path)

def ask_prediction(username:str, password:str, sentence_to_predict:str, model_name):
    predicted_class = ""
    return predicted_class

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
        'universal-sentence-encoder-multilingual'
        'universal-sentence-encoder-multilingual-large'
    ]

    return available_models
