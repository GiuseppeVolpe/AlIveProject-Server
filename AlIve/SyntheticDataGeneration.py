import os
import pandas as pd
from tqdm import tqdm

def compose_begins(greetings, second_half_begins):        

    sentence_begins = []

    for greeting in greetings:
        for second_half_begin in second_half_begins:
            if greeting != "":
                sentence_begins.append([(greeting, "O")] + second_half_begin)
            else:
                sentence_begins.append(second_half_begin)
    
    return sentence_begins

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
                                [("bel", "O"), ("cucciolino", "O")],
                                [("bello", "O")]
                            ])

#region Entities

the_bush = [
    [("questo", "O"), ("cespuglio", "B-Placeholder")], 
    [("quel", "O"), ("cespuglio", "B-Placeholder")], 
    [("il", "O"), ("cespuglio", "B-Placeholder")], 
    [("questo", "O"), ("cespo", "B-Placeholder")], 
    [("quel", "O"), ("cespo", "B-Placeholder")], 
    [("il", "O"), ("cespo", "B-Placeholder")], 
]

to_the_bush = [
    [("a", "O"), ("questo", "O"), ("cespuglio", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("cespuglio", "B-Placeholder")], 
    [("al", "O"), ("cespuglio", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("cespo", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("cespo", "B-Placeholder")], 
    [("al", "O"), ("cespo", "B-Placeholder")], 
]

in_the_bush = [
    [("in", "O"), ("questo", "O"), ("cespuglio", "B-Placeholder")], 
    [("in", "O"), ("quel", "O"), ("cespuglio", "B-Placeholder")], 
    [("nel", "O"), ("cespuglio", "B-Placeholder")], 
    [("in", "O"), ("questo", "O"), ("cespo", "B-Placeholder")], 
    [("in", "O"), ("quel", "O"), ("cespo", "B-Placeholder")], 
    [("nel", "O"), ("cespo", "B-Placeholder")], 
]

the_ball = [
    [("questa", "O"), ("palla", "B-Placeholder")], 
    [("quella", "O"), ("palla", "B-Placeholder")], 
    [("la", "O"), ("palla", "B-Placeholder")], 
    [("questa", "O"), ("sfera", "B-Placeholder")], 
    [("quella", "O"), ("sfera", "B-Placeholder")], 
    [("la", "O"), ("sfera", "B-Placeholder")], 
    [("questo", "O"), ("pallone", "B-Placeholder")], 
    [("quel", "O"), ("pallone", "B-Placeholder")], 
    [("il", "O"), ("pallone", "B-Placeholder")], 
]

to_the_ball = [
    [("a", "O"), ("questa", "O"), ("palla", "B-Placeholder")], 
    [("a", "O"), ("quella", "O"), ("palla", "B-Placeholder")], 
    [("alla", "O"), ("palla", "B-Placeholder")], 
    [("a", "O"), ("questa", "O"), ("sfera", "B-Placeholder")], 
    [("a", "O"), ("quella", "O"), ("sfera", "B-Placeholder")], 
    [("alla", "O"), ("sfera", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("pallone", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("pallone", "B-Placeholder")], 
    [("al", "O"), ("pallone", "B-Placeholder")], 
]

the_fire = [
    [("questo", "O"), ("fuoco", "B-Placeholder")], 
    [("quel", "O"), ("fuoco", "B-Placeholder")], 
    [("il", "O"), ("fuoco", "B-Placeholder")], 
    [("questo", "O"), ("falò", "B-Placeholder")], 
    [("quel", "O"), ("falò", "B-Placeholder")], 
    [("il", "O"), ("falò", "B-Placeholder")], 
    [("questo", "O"), ("falo", "B-Placeholder")], 
    [("quel", "O"), ("falo", "B-Placeholder")], 
    [("il", "O"), ("falo", "B-Placeholder")], 
    [("questo", "O"), ("fuocherello", "B-Placeholder")], 
    [("quel", "O"), ("fuocherello", "B-Placeholder")], 
    [("il", "O"), ("fuocherello", "B-Placeholder")], 
    [("questa", "O"), ("fiamma", "B-Placeholder")], 
    [("quella", "O"), ("fiamma", "B-Placeholder")], 
    [("la", "O"), ("fiamma", "B-Placeholder")], 
]

to_the_fire = [
    [("a", "O"), ("questo", "O"), ("fuoco", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("fuoco", "B-Placeholder")], 
    [("al", "O"), ("fuoco", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("falò", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("falò", "B-Placeholder")], 
    [("al", "O"), ("falò", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("falo", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("falo", "B-Placeholder")], 
    [("al", "O"), ("falo", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("fuocherello", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("fuocherello", "B-Placeholder")], 
    [("al", "O"), ("fuocherello", "B-Placeholder")], 
    [("a", "O"), ("questa", "O"), ("fiamma", "B-Placeholder")], 
    [("a", "O"), ("quella", "O"), ("fiamma", "B-Placeholder")], 
    [("alla", "O"), ("fiamma", "B-Placeholder")], 
]

the_key = [
    [("questa", "O"), ("chiave", "B-Placeholder")], 
    [("quella", "O"), ("chiave", "B-Placeholder")], 
    [("la", "O"), ("chiave", "B-Placeholder")], 
    [("queste", "O"), ("chiavi", "B-Placeholder")], 
    [("quelle", "O"), ("chiavi", "B-Placeholder")], 
    [("le", "O"), ("chiavi", "B-Placeholder")], 
]

to_the_key = [
    [("a", "O"), ("questa", "O"), ("chiave", "B-Placeholder")], 
    [("a", "O"), ("quella", "O"), ("chiave", "B-Placeholder")], 
    [("alla", "O"), ("chiave", "B-Placeholder")], 
    [("a", "O"), ("queste", "O"), ("chiavi", "B-Placeholder")], 
    [("a", "O"), ("quelle", "O"), ("chiavi", "B-Placeholder")], 
    [("alle", "O"), ("chiavi", "B-Placeholder")], 
]

the_log = [
    [("questo", "O"), ("ramo", "B-Placeholder")], 
    [("quel", "O"), ("ramo", "B-Placeholder")], 
    [("il", "O"), ("ramo", "B-Placeholder")], 
    [("questo", "O"), ("ramoscello", "B-Placeholder")], 
    [("quel", "O"), ("ramoscello", "B-Placeholder")], 
    [("il", "O"), ("ramoscello", "B-Placeholder")], 
    [("questo", "O"), ("rametto", "B-Placeholder")], 
    [("quel", "O"), ("rametto", "B-Placeholder")], 
    [("il", "O"), ("rametto", "B-Placeholder")], 
    [("questo", "O"), ("bastone", "B-Placeholder")], 
    [("quel", "O"), ("bastone", "B-Placeholder")], 
    [("il", "O"), ("bastone", "B-Placeholder")], 
    [("questa", "O"), ("bacchetta", "B-Placeholder")], 
    [("quella", "O"), ("bacchetta", "B-Placeholder")], 
    [("la", "O"), ("bacchetta", "B-Placeholder")], 
]

to_the_log = [
    [("a", "O"), ("questo", "O"), ("ramo", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("ramo", "B-Placeholder")], 
    [("al", "O"), ("ramo", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("ramoscello", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("ramoscello", "B-Placeholder")], 
    [("al", "O"), ("ramoscello", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("rametto", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("rametto", "B-Placeholder")], 
    [("al", "O"), ("rametto", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("bastone", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("bastone", "B-Placeholder")], 
    [("al", "O"), ("bastone", "B-Placeholder")], 
    [("a", "O"), ("questa", "O"), ("bacchetta", "B-Placeholder")], 
    [("a", "O"), ("quella", "O"), ("bacchetta", "B-Placeholder")], 
    [("alla", "O"), ("bacchetta", "B-Placeholder")], 
]

the_pot = [
    [("questo", "O"), ("vaso", "B-Placeholder")], 
    [("quel", "O"), ("vaso", "B-Placeholder")], 
    [("il", "O"), ("vaso", "B-Placeholder")], 
    [("questa", "O"), ("pianta", "B-Placeholder")], 
    [("quella", "O"), ("pianta", "B-Placeholder")], 
    [("la", "O"), ("pianta", "B-Placeholder")], 
    [("questa", "O"), ("pianta", "B-Placeholder"), ("nel", "I-Placeholder"), ("vaso", "I-Placeholder")], 
    [("quella", "O"), ("pianta", "B-Placeholder"), ("nel", "I-Placeholder"), ("vaso", "I-Placeholder")], 
    [("la", "O"), ("pianta", "B-Placeholder"), ("nel", "I-Placeholder"), ("vaso", "I-Placeholder")], 
]

to_the_pot = [
    [("a", "O"), ("questo", "O"), ("vaso", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("vaso", "B-Placeholder")], 
    [("al", "O"), ("vaso", "B-Placeholder")], 
    [("a", "O"), ("questa", "O"), ("pianta", "B-Placeholder")], 
    [("a", "O"), ("quella", "O"), ("pianta", "B-Placeholder")], 
    [("alla", "O"), ("pianta", "B-Placeholder")], 
    [("a", "O"), ("questa", "O"), ("pianta", "B-Placeholder"), ("nel", "I-Placeholder"), ("vaso", "I-Placeholder")], 
    [("a", "O"), ("quella", "O"), ("pianta", "B-Placeholder"), ("nel", "I-Placeholder"), ("vaso", "I-Placeholder")], 
    [("alla", "O"), ("pianta", "B-Placeholder"), ("nel", "I-Placeholder"), ("vaso", "I-Placeholder")], 
]

in_the_pot = [
    [("in", "O"), ("questo", "O"), ("vaso", "B-Placeholder")], 
    [("in", "O"), ("quel", "O"), ("vaso", "B-Placeholder")], 
    [("nel", "O"), ("vaso", "B-Placeholder")], 
    [("in", "O"), ("questa", "O"), ("pianta", "B-Placeholder")], 
    [("in", "O"), ("quella", "O"), ("pianta", "B-Placeholder")], 
    [("nella", "O"), ("pianta", "B-Placeholder")], 
    [("in", "O"), ("questa", "O"), ("pianta", "B-Placeholder"), ("nel", "I-Placeholder"), ("vaso", "I-Placeholder")], 
    [("in", "O"), ("quella", "O"), ("pianta", "B-Placeholder"), ("nel", "I-Placeholder"), ("vaso", "I-Placeholder")], 
    [("nella", "O"), ("pianta", "B-Placeholder"), ("nel", "I-Placeholder"), ("vaso", "I-Placeholder")], 
]

the_bone = [
    [("questo", "O"), ("osso", "B-Placeholder")], 
    [("quell", "O"), ("osso", "B-Placeholder")], 
    [("l", "O"), ("osso", "B-Placeholder")], 
    [("questo", "O"), ("gioco", "B-Placeholder")], 
    [("quel", "O"), ("gioco", "B-Placeholder")], 
    [("il", "O"), ("gioco", "B-Placeholder")], 
    [("questo", "O"), ("giochino", "B-Placeholder")], 
    [("quel", "O"), ("giochino", "B-Placeholder")], 
    [("il", "O"), ("giochino", "B-Placeholder")], 
    [("questo", "O"), ("giochetto", "B-Placeholder")], 
    [("quel", "O"), ("giochetto", "B-Placeholder")], 
    [("il", "O"), ("giochetto", "B-Placeholder")], 
]

to_the_bone = [
    [("a", "O"), ("questo", "O"), ("osso", "B-Placeholder")], 
    [("a", "O"), ("quell", "O"), ("osso", "B-Placeholder")], 
    [("all", "O"), ("osso", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("gioco", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("gioco", "B-Placeholder")], 
    [("al", "O"), ("gioco", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("giochino", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("giochino", "B-Placeholder")], 
    [("al", "O"), ("giochino", "B-Placeholder")], 
    [("a", "O"), ("questo", "O"), ("giochetto", "B-Placeholder")], 
    [("a", "O"), ("quel", "O"), ("giochetto", "B-Placeholder")], 
    [("al", "O"), ("giochetto", "B-Placeholder")], 
]

the_doghouse = [
    [("questa", "O"), ("cuccia", "B-Placeholder")], 
    [("quella", "O"), ("cuccia", "B-Placeholder")], 
    [("la", "O"), ("cuccia", "B-Placeholder")], 
    [("questa", "O"), ("casetta", "B-Placeholder")], 
    [("quella", "O"), ("casetta", "B-Placeholder")], 
    [("la", "O"), ("casetta", "B-Placeholder")],
]

to_the_doghouse = [
    [("a", "O"), ("questa", "O"), ("cuccia", "B-Placeholder")], 
    [("a", "O"), ("quella", "O"), ("cuccia", "B-Placeholder")], 
    [("alla", "O"), ("cuccia", "B-Placeholder")], 
    [("a", "O"), ("questa", "O"), ("casetta", "B-Placeholder")], 
    [("a", "O"), ("quella", "O"), ("casetta", "B-Placeholder")], 
    [("alla", "O"), ("casetta", "B-Placeholder")],
]

in_the_doghouse = [
    [("in", "O"), ("questa", "O"), ("cuccia", "B-Placeholder")], 
    [("in", "O"), ("quella", "O"), ("cuccia", "B-Placeholder")], 
    [("nella", "O"), ("cuccia", "B-Placeholder")], 
    [("in", "O"), ("questa", "O"), ("casetta", "B-Placeholder")], 
    [("in", "O"), ("quella", "O"), ("casetta", "B-Placeholder")], 
    [("nella", "O"), ("casetta", "B-Placeholder")],
]

the_reachables = the_bush + the_ball + the_fire + the_key + the_log + the_pot + the_bone + the_doghouse
the_pickables = the_ball + the_key + the_log + the_bone
the_throwables = the_ball + the_key + the_log + the_bone
the_hittables = the_bush + the_ball + the_key + the_log + the_pot + the_bone + the_doghouse
the_inspectables = the_bush + the_doghouse

to_the_reachables = to_the_bush + to_the_ball + to_the_fire + to_the_key + to_the_log + to_the_pot + to_the_bone + to_the_doghouse
to_the_pickables = to_the_ball + to_the_key + to_the_log + to_the_bone
to_the_throwables = to_the_ball + to_the_key + to_the_log + to_the_bone
to_the_hittables = to_the_bush + to_the_ball + to_the_key + to_the_log + to_the_pot + to_the_bone + to_the_doghouse
to_the_inspectables = to_the_bush + to_the_doghouse

in_the_reachables = in_the_bush + in_the_pot + in_the_doghouse
in_the_hittables = in_the_bush + in_the_pot + in_the_doghouse
in_the_inspectables = in_the_bush + in_the_doghouse

#endregion

sentence_communicative_intents_positive = {
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
}

for the_reachable in the_reachables:

    temporary_reachable = list()
    
    for t in the_reachable:
        new_word = (t[0], t[1])
        temporary_reachable.append(new_word)
    
    for i, word in enumerate(temporary_reachable):
        if word[1] == "B-Placeholder":
            temporary_reachable[i] = (word[0], "B-ToReach")
        if word[1] == "I-Placeholder":
            temporary_reachable[i] = (word[0], "I-ToReach")

    sentence_communicative_intents_positive["Reach"].append(
        [("raggiungi", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("vai", "O"), ("dove", "O"), ("sta", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("vai", "O"), ("dove", "O"), ("si", "O"), ("trova", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("vai", "O"), ("in", "O"), ("corrispondenza", "O"), ("di", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("recati", "O"), ("dove", "O"), ("sta", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("recati", "O"), ("dove", "O"), ("si", "O"), ("trova", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("recati", "O"), ("in", "O"), ("corrispondenza", "O"), ("di", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("andresti", "O"), ("dove", "O"), ("sta", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("andresti", "O"), ("dove", "O"), ("si", "O"), ("trova", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("andresti", "O"), ("in", "O"), ("corrispondenza", "O"), ("di", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("ti", "O"), ("recheresti", "O"), ("dove", "O"), ("sta", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("ti", "O"), ("recheresti", "O"), ("dove", "O"), ("si", "O"), ("trova", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("ti", "O"), ("recheresti", "O"), ("in", "O"), ("corrispondenza", "O"), ("di", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("puoi", "O"), ("andare", "O"), ("dove", "O"), ("sta", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("puoi", "O"), ("andare", "O"), ("dove", "O"), ("si", "O"), ("trova", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("puoi", "O"), ("andare", "O"), ("in", "O"), ("corrispondenza", "O"), ("di", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("potresti", "O"), ("andare", "O"), ("dove", "O"), ("sta", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("potresti", "O"), ("andare", "O"), ("dove", "O"), ("si", "O"), ("trova", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("potresti", "O"), ("andare", "O"), ("in", "O"), ("corrispondenza", "O"), ("di", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("puoi", "O"), ("recarti", "O"), ("dove", "O"), ("sta", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("puoi", "O"), ("recarti", "O"), ("dove", "O"), ("si", "O"), ("trova", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("puoi", "O"), ("recarti", "O"), ("in", "O"), ("corrispondenza", "O"), ("di", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("potresti", "O"), ("recarti", "O"), ("dove", "O"), ("sta", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("potresti", "O"), ("recarti", "O"), ("dove", "O"), ("si", "O"), ("trova", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("potresti", "O"), ("recarti", "O"), ("in", "O"), ("corrispondenza", "O"), ("di", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("puoi", "O"), ("raggiungere", "O")] + temporary_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("potresti", "O"), ("raggiungere", "O")] + temporary_reachable
    )

for to_the_reachable in to_the_reachables:

    temporary_to_the_reachable = list()
    
    for t in to_the_reachable:
        new_word = (t[0], t[1])
        temporary_to_the_reachable.append(new_word)
    
    for i, word in enumerate(temporary_to_the_reachable):
        if word[1] == "B-Placeholder":
            temporary_to_the_reachable[i] = (word[0], "B-ToReach")
        if word[1] == "I-Placeholder":
            temporary_to_the_reachable[i] = (word[0], "I-ToReach")
    
    sentence_communicative_intents_positive["Reach"].append(
        [("vai", "O"), ("vicino", "O")] + temporary_to_the_reachable
    )
    
    sentence_communicative_intents_positive["Reach"].append(
        [("andresti", "O"), ("vicino", "O")] + temporary_to_the_reachable
    )

for the_pickable in the_pickables:

    temporary_the_pickable = list()
    
    for t in the_pickable:
        new_word = (t[0], t[1])
        temporary_the_pickable.append(new_word)
    
    for i, word in enumerate(temporary_the_pickable):
        if word[1] == "B-Placeholder":
            temporary_the_pickable[i] = (word[0], "B-ToPick")
        if word[1] == "I-Placeholder":
            temporary_the_pickable[i] = (word[0], "I-ToPick")

    sentence_communicative_intents_positive["Pick"].append(
        [("raccogli", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("raccoglieresti", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("puoi", "O"), ("raccogliere", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("potresti", "O"), ("raccogliere", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("prendi", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("prenderesti", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("puoi", "O"), ("prendere", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("potresti", "O"), ("prendere", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("afferra", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("affereresti", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("puoi", "O"), ("afferrare", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("potresti", "O"), ("afferrare", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("agguanta", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("agguanteresti", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("puoi", "O"), ("agguantare", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Pick"].append(
        [("potresti", "O"), ("agguantare", "O")] + temporary_the_pickable
    )

for the_pickable in the_pickables:

    temporary_the_pickable = list()
    
    for t in the_pickable:
        new_word = (t[0], t[1])
        temporary_the_pickable.append(new_word)
    
    for i, word in enumerate(temporary_the_pickable):
        if word[1] == "B-Placeholder":
            temporary_the_pickable[i] = (word[0], "B-ToLeave")
        if word[1] == "I-Placeholder":
            temporary_the_pickable[i] = (word[0], "I-ToLeave")

    sentence_communicative_intents_positive["Leave"].append(
        [("lascia", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("lasceresti", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("puoi", "O"), ("lasciare", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("potresti", "O"), ("lasciare", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("molla", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("molleresti", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("puoi", "O"), ("mollare", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("potresti", "O"), ("mollare", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("getta", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("butta", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("getteresti", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("butteresti", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("puoi", "O"), ("gettare", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("potresti", "O"), ("gettare", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("rilascia", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("rilasceresti", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("puoi", "O"), ("rilasciare", "O")] + temporary_the_pickable
    )
    
    sentence_communicative_intents_positive["Leave"].append(
        [("potresti", "O"), ("rilasciare", "O")] + temporary_the_pickable
    )

for the_pickable in the_pickables:

    temporary_the_pickable = list()
    
    for t in the_pickable:
        new_word = (t[0], t[1])
        temporary_the_pickable.append(new_word)
    
    for i, word in enumerate(temporary_the_pickable):
        if word[1] == "B-Placeholder":
            temporary_the_pickable[i] = (word[0], "B-ToThrow")
        if word[1] == "I-Placeholder":
            temporary_the_pickable[i] = (word[0], "I-ToThrow")
    
    for the_hittable in the_hittables:

        temporary_the_hittable = list()
        
        for t in the_hittable:
            new_word = (t[0], t[1])
            temporary_the_hittable.append(new_word)
        
        for i, word in enumerate(the_hittable):
            if word[1] == "B-Placeholder":
                temporary_the_hittable[i] = (word[0], "B-ToHit")
            if word[1] == "I-Placeholder":
                temporary_the_hittable[i] = (word[0], "I-ToHit")
        
        sentence_communicative_intents_positive["Throw"].append(
            [("colpisci", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("colpisci", "O")] + temporary_the_hittable + [("con", "O")] + temporary_the_pickable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("colpisci", "O")] + temporary_the_hittable + [("usando", "O")] + temporary_the_pickable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("lancia", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("scaglia", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("scaraventa", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("puoi", "O"), ("colpire", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("puoi", "O"), ("colpire", "O")] + temporary_the_hittable + [("con", "O")] + temporary_the_pickable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("puoi", "O"), ("colpire", "O")] + temporary_the_hittable + [("usando", "O")] + temporary_the_pickable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("puoi", "O"), ("lanciare", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("puoi", "O"), ("scagliare", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("puoi", "O"), ("scaraventare", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("potresti", "O"), ("colpire", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("potresti", "O"), ("colpire", "O")] + temporary_the_hittable + [("con", "O")] + temporary_the_pickable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("potresti", "O"), ("colpire", "O")] + temporary_the_hittable + [("usando", "O")] + temporary_the_pickable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("potresti", "O"), ("lanciare", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("potresti", "O"), ("scagliare", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("potresti", "O"), ("scaraventare", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("colpiresti", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("lanceresti", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("scaglieresti", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )
        
        sentence_communicative_intents_positive["Throw"].append(
            [("scaraventeresti", "O")] + temporary_the_pickable + [("contro", "O")] + temporary_the_hittable
        )

for the_inspectable in the_inspectables:

    temporary_the_inspectable = list()
    
    for t in the_inspectable:
        new_word = (t[0], t[1])
        temporary_the_inspectable.append(new_word)
    
    for i, word in enumerate(temporary_the_inspectable):
        if word[1] == "B-Placeholder":
            temporary_the_inspectable[i] = (word[0], "B-ToInspect")
        if word[1] == "I-Placeholder":
            temporary_the_inspectable[i] = (word[0], "I-ToInspect")

    sentence_communicative_intents_positive["Inspect"].append(
        [("controlla", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("ispeziona", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("esamina", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("vedi", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("analizza", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("puoi", "O"), ("controllare", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("puoi", "O"), ("ispezionare", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("puoi", "O"), ("esaminare", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("puoi", "O"), ("vedere", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("puoi", "O"), ("analizzare", "O")] + temporary_the_inspectable
    )

    sentence_communicative_intents_positive["Inspect"].append(
        [("potresti", "O"), ("controllare", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("potresti", "O"), ("ispezionare", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("potresti", "O"), ("esaminare", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("potresti", "O"), ("vedere", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("potresti", "O"), ("analizzare", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("controlleresti", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("ispezioneresti", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("esamineresti", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("vedresti", "O")] + temporary_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("analizzerersti", "O")] + temporary_the_inspectable
    )

for to_the_inspectable in to_the_inspectables:

    temporary_to_the_inspectable = list()
    
    for t in to_the_inspectable:
        new_word = (t[0], t[1])
        temporary_to_the_inspectable.append(new_word)
    
    for i, word in enumerate(temporary_to_the_inspectable):
        if word[1] == "B-Placeholder":
            temporary_to_the_inspectable[i] = (word[0], "B-ToInspect")
        if word[1] == "I-Placeholder":
            temporary_to_the_inspectable[i] = (word[0], "I-ToInspect")

    sentence_communicative_intents_positive["Inspect"].append(
        [("dai", "O"), ("un", "O"), ("occhiata", "O")] + temporary_to_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("puoi", "O"), ("dare", "O"), ("un", "O"), ("occhiata", "O")] + temporary_to_the_inspectable
    )

    sentence_communicative_intents_positive["Inspect"].append(
        [("potresti", "O"), ("dare", "O"), ("un", "O"), ("occhiata", "O")] + temporary_to_the_inspectable
    )
    
    sentence_communicative_intents_positive["Inspect"].append(
        [("daresti", "O"), ("un", "O"), ("occhiata", "O")] + temporary_to_the_inspectable
    )

sentence_end_positive = [
                            [("grazie", "O")], 
                            [("per", "O"), ("favore", "O")],
                            [("per", "O"), ("cortesia", "O")],
                            [("su", "O"), ("bello", "O")], 
                            [("dai", "O"), ("bello", "O")],
                            [("puoi", "O"), ("farcela", "O")],
                            [("fallo", "O"), ("per", "O"), ("me", "O")],
                        ]

built_sentences = list()

ic_dataframe = pd.DataFrame({"text":[], "intent":[]})
ner_dataframe = pd.DataFrame({"sentence_idx":[], "word":[], "tag":[]})

category = "train"
ic_file_name = "synthetic_dataset_ic_" + category
ner_file_name = "synthetic_dataset_ner_" + category
ic_file_path = ic_file_name + ".csv"
ner_file_path = ner_file_name + ".csv"

sentence_idx = 0

for intent in sentence_communicative_intents_positive.keys():
    for sentence_com_in in tqdm(sentence_communicative_intents_positive[intent]):
            sentence_idx += 1
            final_sentence = sentence_com_in
            built_sentence = ""

            for word in final_sentence:

                ner_dataframe.loc[len(ner_dataframe)] = {"sentence_idx" : sentence_idx, 
                                                            "word": word[0],
                                                            "tag": word[1]}

                if len(word) > 2:
                    input(word)

                built_sentence += word[0] + " "
            
            ic_dataframe.loc[len(ic_dataframe)] = {"text": built_sentence, "intent": intent}

ic_dataframe.to_csv(ic_file_path)
ner_dataframe.to_csv(ner_file_path)
