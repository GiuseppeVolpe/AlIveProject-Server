import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

def split_data(df:pd.DataFrame, text_column_name:str, label_column_name:str, 
               train_percentage:float=0.7, valid_percentage:float=0.15):
    
    if(text_column_name not in df or label_column_name not in df):
        raise Exception("Dataframe malformed!")
    
    train_percentage = clamp(train_percentage, 0.7, 0.9)
    remained_percentage = 1 - train_percentage
    valid_percentage = clamp(valid_percentage, remained_percentage / 2, remained_percentage / 1.5)
    
    train_text, temp_text, train_labels, temp_labels = train_test_split(df[text_column_name], df[label_column_name],
                                                                        random_state = 2021,
                                                                        test_size = 1 - train_percentage,
                                                                        stratify = df[label_column_name])

    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                    random_state = 2021,
                                                                    test_size = 1 - (train_percentage + valid_percentage),
                                                                    stratify = temp_labels)
    
    train = pd.DataFrame({text_column_name: train_text, label_column_name: train_labels})
    validation = pd.DataFrame({text_column_name: val_text, label_column_name: val_labels})
    test = pd.DataFrame({text_column_name: test_text, label_column_name: test_labels})

    return train, validation, test

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def retcon_alive_file(dir:str, toadd:dict=None, toremove:list=None, alive_file_name="alive_data.pickle"):
    
    if toadd == None:
        toadd = dict()
    
    if toremove == None:
        toremove = list()

    dir = dir.replace("\\", "/")

    if dir[-1] == "/":
        dir = dir[:-1]
    
    alive_data_path = dir + "/" + alive_file_name

    retconned_data = dict()

    with open(alive_data_path, "rb") as infile:
        alive_data = pickle.load(infile)

        for key in alive_data.keys():
            if key not in toremove:
                retconned_data[key] = alive_data[key]
    
    for key in toadd.keys():
        retconned_data[key] = toadd[key]
    
    try:
        if os.path.exists(alive_data_path):
            os.remove(alive_data_path)
        outfile = open(alive_data_path, "wb")
        pickle.dump(retconned_data, outfile)
        outfile.close()
    except:
        raise Exception("Couldn't save the corrected AlIve file!")
