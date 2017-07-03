from PIL import Image
from scipy import misc
import numpy as np
import os, os.path

i=1
for filename in os.listdir("."):
    #os.rename(filename,str(i)+".jpg")


    image = misc.imread(str(i)+".jpg")

    def weightedAverage(pixel):
        return 0.299*pixel[0]+0.587*pixel[1]+0.114*pixel[2]

    grey = np.zeros((image.shape[0],image.shape[1]))
    #get row number
    for rownum in range(len(image)):
        for colnum in range(len(image)):
            grey[rownum][colnum] = weightedAverage(image[rownum][colnum])


    misc.imsave(str(i)+"grey.jpg", grey)

    img = Image.open(str(i)+"grey.jpg")

    imagetopleft = img.crop((0,0,49,49))
    imagetopleft.save(str(i)+"greytl.jpg")


    imagetopright = img.crop((1,0,50,49))
    imagetopright.save(str(i)+"greytr.jpg")

    imagebotleft = img.crop((0,1,49,50))
    imagebotleft.save(str(i)+"greybl.jpg")

    imagebotright = img.crop((1,1,50,50))
    imagebotright.save(str(i)+"greybr.jpg")

    i+= 1



