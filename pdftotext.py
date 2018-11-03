from tika import parser
import os, sys


f = open("test.txt", "wb")
raw = parser.from_file(os.listdir(os.getcwd()))
print(raw['content'])
f.write(raw['content'].encode('utf-8'))
f.close()
