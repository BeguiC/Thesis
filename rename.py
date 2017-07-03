import os, os.path

i=1
for filename in os.listdir("."):
    os.rename(filename,str(i)+".jpg")
    i+=1
