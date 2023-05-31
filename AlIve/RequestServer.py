#!/usr/bin/env python3

# import socket programming library
import socket
import pickle

# import thread module
from _thread import *
import threading

from ModelsAndDatasets import *

print_lock = threading.Lock()

# thread function
def threaded(c):

	loaded_models = dict()

	while True:

		# data received from client
		data = c.recv(1024)
		if not data:
			print('Bye')
			
			# lock released on exit
			print_lock.release()
			break

		# reverse the given string from client
		alive_request = pickle.loads(data)

		if alive_request["model"] not in loaded_models.keys():
			path_to_model = "AlIve/UsersData/" + alive_request["user"] + "/Ambients/Models/" + alive_request["model"] + "/"
			loaded_model = NLPClassificationModel.load_model(path_to_model)
			loaded_models[alive_request["model"]] = loaded_model
		
		desired_model = loaded_models[alive_request["model"]]

		prediction = desired_model.predict([alive_request["sentence"]])

		response = pickle.dumps(prediction)

		# send back reversed string to client
		c.send(response)

	# connection closed
	c.close()

def Main():
	host = ""

	# reserve a port on your computer
	# in our case it is 12345 but it
	# can be anything
	port = 12345
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.bind((host, port))
	print("socket binded to port", port)

	# put the socket into listening mode
	s.listen(5)
	print("socket is listening")

	# a forever loop until client wants to exit
	while True:

		# establish connection with client
		c, addr = s.accept()

		# lock acquired by client
		print_lock.acquire()
		print('Connected to :', addr[0], ':', addr[1])

		# Start a new thread and return its identifier
		start_new_thread(threaded, (c,))
	
	s.close()

if __name__ == '__main__':
	Main()
