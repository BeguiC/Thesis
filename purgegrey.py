import os, os.path

i=1
for filename in os.listdir("."):
    
    os.remove(str(i)+"grey.jpg")
    i+=1
