import os
import openai
import pandas as pd

user_types = {"Un bambino di cinque anni", 
              "Un ragazzino di quindici anni", 
              "Un ragazzo di venti anni", 
              "Un uomo di quaranta anni", 
              "Un ingegnere molto intelligente e severo",
              "Un ragazzo espansivo e simpatico",
              "Un uomo molto rude"}

actions = {
    "MoveToRoom" : ["Spostarsi nella stanza a sinistra", "Spostarsi nella stanza a destra", 
                    "Spostarsi nella stanza a sud", "Spostarsi nella stanza a nord"], 
    "GoDownstairs" : ["Scendere le scale", "Andare giù per le scale"], 
    "GoUpstairs" : ["Salire le scale", "Andare su per le scale"], 
    "OpenContainer" : ["Aprire un cassetto", "Aprire un armadio", 
                       "Aprire una cassetta", "Aprire un contenitore", 
                       "Aprire una scatola", "Sbirciare in un cassetto", 
                       "Sbirciare in un armadio", "Sbirciare in una cassetta", 
                       "Sbirciare in un contenitore", "Sbirciare in una scatola", 
                       "Aprire un armadietto", "Sbirciare in un armadietto"], 
    "PickItem" : ["Prendere un oggetto", "Raccogliere qualcosa"], 
    "UseDevice" : ["Usare un computer", "Usare un laptop", "Usare un telefono"], 
    "Read" : ["Leggere un documento", "Leggere un file"], 
    "Write" : ["Scrivere", "Inserire un codice"], 
    "UseKeyItem" : ["Inserire l'oggetto", "Usare la chiave"], 
    "Examine" : ["Esaminare un oggetto", "Studiare un oggetto", "Controllare un oggetto", "Analizzare un oggetto"]
}

user_moods = {"In maniera molto gentile", "In maniera neutrale", "Senza alcuna gentilezza", "In maniera fredda e distaccata"}

agents = {"Un bambino", 
          "Un ragazzo", 
          "Un uomo", 
          "Una bambina", 
          "Una ragazza", 
          "Una donna"}

type_of_requests = {"la richiesta", "l'ordine", "il comando"}

temperatures = [0.6, 0.7, 0.8, 0.9, 1]

num_of_actions = 0
for intent in actions.keys():
    num_of_actions += len(actions[intent])

input("Number of possibilities: {}.".format(len(user_types) * num_of_actions * len(user_moods) * len(agents) * len(temperatures) * len(type_of_requests)))

prompt = "Devi chiedere a <<AGENT>> di <<ACTION>>, <<MOOD>>. Come formuleresti <<REQUEST TYPE>>"

openai.api_key = "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"

path = "synthetic_dataset.csv"

if os.path.exists(path):
    dataframe = pd.read_csv(path)
else:
    dataframe = pd.DataFrame({"text":[], "intent":[]})

text_dict = dict()

for user in user_types:
    for intent in actions.keys():
        for action in actions[intent]:
            for agent in agents:
                for mood in user_moods:
                    for request_type in type_of_requests:
                        for temperature in temperatures:
                            role = "Sei " + user
                            content = prompt.replace("<<AGENT>>", agent).replace("<<ACTION>>", action).replace("<<MOOD>>", mood).replace("<<REQUEST TYPE>>", request_type)
                            
                            attempts = 0

                            for attempt in range(10):
                                
                                try:
                                    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                                                                messages=[{"role": "system", "content": role}, 
                                                                                            {"role": "user", "content": content}], 
                                                                                temperature=temperature, 
                                                                                max_tokens=50)
                                    break
                                except Exception as ex:
                                    print(ex)

                            for choice in chat_completion.choices:
                                response = choice.message.content

                                if response[0] == '"' and response[-1] == '"':

                                    text = response.replace('"', "")
                                    
                                    if text not in text_dict.keys():

                                        text_dict[text] = True

                                        new_row = {
                                            "text" : text, 
                                            "intent" : intent
                                        }

                                        dataframe.loc[len(dataframe)] = new_row

                                        if os.path.exists(path):
                                            os.remove(path)
                                        
                                        dataframe.to_csv(path)
