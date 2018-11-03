from tika import parser
import os, sys

#print(os.listdir(os.getcwd()))

#f = open("test.txt", "wb")
#raw = parser.from_file(os.listdir(os.getcwd()))
#print(raw['content'])
#f.write(raw['content'].encode('utf-8'))
#f.close()

files = os.listdir(os.getcwd())
for x in range (0, len(files)):
	raw = parser.from_file(files[x])
	print(raw['content'])
	f = open(files[x]+".txt", "wb")
	f.write(raw['content'].encode('utf-8'))
	f.close()