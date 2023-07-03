import os
import pandas as pd

category = "train"
ic_file_name = "synthetic_dataset_ic_" + category
ner_file_name = "synthetic_dataset_ner_" + category
ic_file_path = ic_file_name + ".csv"
ner_file_path = ner_file_name + ".csv"

built_sentences = list()

ic_dataframe = pd.DataFrame({"text":[], "intent":[]})
ner_dataframe = pd.DataFrame({"sentence_idx":[], "word":[], "tag":[]})

ic_dataframe.to_csv(ic_file_path)
ner_dataframe.to_csv(ner_file_path)
