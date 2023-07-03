import os
import pandas as pd

def compose_begins(greetings, second_half_begins):        

    sentence_begins = []

    for greeting in greetings:
        for second_half_begin in second_half_begins:
            if greeting != "":
                sentence_begins.append([(greeting, "O")] + second_half_begin)
            else:
                sentence_begins.append(second_half_begin)

sentence_begins_positive = compose_begins(
                            ["", "hey", "ciao", "salve", "buongiorno", "buonasera", "hola", "heila"],
                            [
                                [("cane", "O")], 
                                [("bel", "O"), ("cane", "O")],
                                [("cagnolino", "O")],
                                [("bel", "O"), ("cagnolino", "O")],
                                [("cucciolo", "O")],
                                [("bel", "O"), ("cucciolo", "O")],
                                [("cucciolino", "O")],
                                [("bel", "O"), ("cucciolino", "O")]
                                [("bello", "O")],
                            ])

sentence_begins_negative = compose_begins(
                            ["oh", "hey"],
                            [
                                [("cane", "O")], 
                                [("cattivo", "O"), ("cane", "O")], 
                                [("stupido", "O"), ("cane", "O")], 
                                [("maledetto", "O"), ("cane", "O")], 
                                [("cane", "O"), ("bastardo", "O")], 
                                [("cane", "O"), ("stronzo", "O")], 
                                [("cane", "O"), ("rognoso", "O")], 
                                [("cane", "O"), ("di", "O"), ("merda", "O")], 
                                [("cane", "O"), ("del", "O"), ("cazzo", "O")], 
                                [("cane", "O"), ("pulcioso", "O")], 
                                [("cagnaccio", "O")], 
                                [("cattivo", "O"), ("cagnaccio", "O")], 
                                [("stupido", "O"), ("cagnaccio", "O")], 
                                [("maledetto", "O"), ("cagnaccio", "O")], 
                                [("cagnaccio", "O"), ("bastardo", "O")], 
                                [("cagnaccio", "O"), ("stronzo", "O")], 
                                [("cagnaccio", "O"), ("rognoso", "O")], 
                                [("cagnaccio", "O"), ("di", "O"), ("merda", "O")], 
                                [("cagnaccio", "O"), ("del", "O"), ("cazzo", "O")], 
                                [("cagnaccio", "O"), ("pulcioso", "O")], 
                                [("bestia", "O")], 
                                [("cattiva", "O"), ("bestia", "O")], 
                                [("stupida", "O"), ("bestia", "O")], 
                                [("maledetta", "O"), ("bestia", "O")], 
                                [("bestia", "O"), ("bastarda", "O")], 
                                [("bestia", "O"), ("stronza", "O")], 
                                [("bestia", "O"), ("rognosa", "O")], 
                                [("bestia", "O"), ("di", "O"), ("merda", "O")], 
                                [("bestia", "O"), ("del", "O"), ("cazzo", "O")], 
                                [("bestia", "O"), ("pulciosa", "O")], 
                                [("bestiaccia", "O")], 
                                [("cattiva", "O"), ("bestiaccia", "O")], 
                                [("stupida", "O"), ("bestiaccia", "O")], 
                                [("maledetta", "O"), ("bestiaccia", "O")], 
                                [("bestiaccia", "O"), ("bastarda", "O")], 
                                [("bestiaccia", "O"), ("stronza", "O")], 
                                [("bestiaccia", "O"), ("rognosa", "O")], 
                                [("bestiaccia", "O"), ("di", "O"), ("merda", "O")], 
                                [("bestiaccia", "O"), ("del", "O"), ("cazzo", "O")], 
                                [("bestiaccia", "O"), ("pulciosa", "O")], 
                                [("sacco", "O"), ("di", "O"), ("pulci", "O")], 
                            ])

#region Entities

reachables = [
    [("questo", "O"), ("cespuglio", "B-ToReach")], 
    [("quel", "O"), ("cespuglio", "B-ToReach")], 
    [("il", "O"), ("cespuglio", "B-ToReach")], 
    
    [("questo", "O"), ("cespo", "B-ToReach")], 
    [("quel", "O"), ("cespo", "B-ToReach")], 
    [("il", "O"), ("cespo", "B-ToReach")], 
    
    [("questa", "O"), ("palla", "B-ToReach")], 
    [("quella", "O"), ("palla", "B-ToReach")], 
    [("la", "O"), ("palla", "B-ToReach")], 
    
    [("questo", "O"), ("pallone", "B-ToReach")], 
    [("quel", "O"), ("pallone", "B-ToReach")], 
    [("il", "O"), ("pallone", "B-ToReach")], 
    
    [("questo", "O"), ("fuoco", "B-ToReach")], 
    [("quel", "O"), ("fuoco", "B-ToReach")], 
    [("il", "O"), ("fuoco", "B-ToReach")], 
    
    [("questo", "O"), ("falò", "B-ToReach")], 
    [("quel", "O"), ("falò", "B-ToReach")], 
    [("il", "O"), ("falò", "B-ToReach")], 
    
    [("questo", "O"), ("falo", "B-ToReach")], 
    [("quel", "O"), ("falo", "B-ToReach")], 
    [("il", "O"), ("falo", "B-ToReach")], 
    
    [("questo", "O"), ("fuocherello", "B-ToReach")], 
    [("quel", "O"), ("fuocherello", "B-ToReach")], 
    [("il", "O"), ("fuocherello", "B-ToReach")], 
    
    [("questa", "O"), ("fiamma", "B-ToReach")], 
    [("quella", "O"), ("fiamma", "B-ToReach")], 
    [("la", "O"), ("fiamma", "B-ToReach")], 

    [("questa", "O"), ("chiave", "B-ToReach")], 
    [("quella", "O"), ("chiave", "B-ToReach")], 
    [("la", "O"), ("chiave", "B-ToReach")], 

    [("queste", "O"), ("chiavi", "B-ToReach")], 
    [("quelle", "O"), ("chiavi", "B-ToReach")], 
    [("le", "O"), ("chiavi", "B-ToReach")], 

    [("questo", "O"), ("ramo", "B-ToReach")], 
    [("quel", "O"), ("ramo", "B-ToReach")], 
    [("il", "O"), ("ramo", "B-ToReach")], 

    [("questo", "O"), ("ramoscello", "B-ToReach")], 
    [("quel", "O"), ("ramoscello", "B-ToReach")], 
    [("il", "O"), ("ramoscello", "B-ToReach")], 

    [("questo", "O"), ("rametto", "B-ToReach")], 
    [("quel", "O"), ("rametto", "B-ToReach")], 
    [("il", "O"), ("rametto", "B-ToReach")], 

    [("questo", "O"), ("bastone", "B-ToReach")], 
    [("quel", "O"), ("bastone", "B-ToReach")], 
    [("il", "O"), ("bastone", "B-ToReach")], 

    [("questa", "O"), ("bacchetta", "B-ToReach")], 
    [("quella", "O"), ("bacchetta", "B-ToReach")], 
    [("la", "O"), ("bacchetta", "B-ToReach")], 

    [("questo", "O"), ("vaso", "B-ToReach")], 
    [("quel", "O"), ("vaso", "B-ToReach")], 
    [("il", "O"), ("vaso", "B-ToReach")], 

    [("questa", "O"), ("pianta", "B-ToReach")], 
    [("quella", "O"), ("pianta", "B-ToReach")], 
    [("la", "O"), ("pianta", "B-ToReach")], 

    [("questa", "O"), ("pianta", "B-ToReach"), ("nel", "I-ToReach"), ("vaso", "I-ToReach")], 
    [("quella", "O"), ("pianta", "B-ToReach"), ("nel", "I-ToReach"), ("vaso", "I-ToReach")], 
    [("la", "O"), ("pianta", "B-ToReach"), ("nel", "I-ToReach"), ("vaso", "I-ToReach")], 

    [("questo", "O"), ("osso", "B-ToReach")], 
    [("quell", "O"), ("osso", "B-ToReach")], 
    [("l", "O"), ("osso", "B-ToReach")], 

    [("questo", "O"), ("gioco", "B-ToReach")], 
    [("quel", "O"), ("gioco", "B-ToReach")], 
    [("il", "O"), ("gioco", "B-ToReach")], 

    [("questo", "O"), ("giochino", "B-ToReach")], 
    [("quel", "O"), ("giochino", "B-ToReach")], 
    [("il", "O"), ("giochino", "B-ToReach")], 

    [("questo", "O"), ("giochetto", "B-ToReach")], 
    [("quel", "O"), ("giochetto", "B-ToReach")], 
    [("il", "O"), ("giochetto", "B-ToReach")], 

    [("questa", "O"), ("cuccia", "B-ToReach")], 
    [("quella", "O"), ("cuccia", "B-ToReach")], 
    [("la", "O"), ("cuccia", "B-ToReach")], 

    [("questa", "O"), ("casetta", "B-ToReach")], 
    [("quella", "O"), ("casetta", "B-ToReach")], 
    [("la", "O"), ("casetta", "B-ToReach")],
]

pickables = []

throwables = []

hittables = []

inspectables = []

#endregion

sentence_communicative_intent_positive = {
    "Reach" : [
        
    ],
    "Pick" : [

    ],
    "Leave" : [

    ],
    "Throw" : [

    ],
    "Inspect" : [

    ],
    "Note" : [

    ],
}

sentence_communicative_intent_negative = {
    "Reach" : [

    ],
    "Pick" : [

    ],
    "Leave" : [

    ],
    "Throw" : [

    ],
    "Inspect" : [

    ],
    "Note" : [

    ],
}

sentence_end_positive = [
                            [("grazie", "O")], 
                            [("per", "O"), ("favore", "O")],
                            [("per", "O"), ("cortesia", "O")],
                            [("su", "O"), ("bello", "O")], 
                            [("dai", "O"), ("bello", "O")],
                            [("puoi", "O"), ("farcela", "O")],
                            [("fallo", "O"), ("per", "O"), ("me", "O")],
                        ]

sentence_end_negative = [
                            [("muoviti", "O")], 
                            [("muoviti", "O"), ("cazzo", "O")],
                            [("dai", "O"), ("cazzo", "O")],
                            [("spicciati", "O")], 
                            [("spicciati", "O"), ("cazzo", "O")],
                            [("datti", "O"), ("una", "O"), ("mossa", "O")],
                            [("sbrigati", "O"), ("cazzo", "O")],
                        ]

built_sentences = list()

ic_dataframe = pd.DataFrame({"text":[], "intent":[]})
ner_dataframe = pd.DataFrame({"sentence_idx":[], "word":[], "tag":[]})

category = "train"
ic_file_name = "synthetic_dataset_ic_" + category
sa_file_name = "synthetic_dataset_sa_" + category
ner_file_name = "synthetic_dataset_ner_" + category
ic_file_path = ic_file_name + ".csv"
sa_file_path = sa_file_name + ".csv"
ner_file_path = ner_file_name + ".csv"

ic_dataframe.to_csv(ic_file_path)
ner_dataframe.to_csv(ner_file_path)
