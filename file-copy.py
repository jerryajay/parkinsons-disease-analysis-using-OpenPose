import os
from shutil import copyfile

path = "/home/jerryant/Desktop/VICON-VIDEO-COMPARISON/"
participant = "Kun-Woo"

copy_to_path = "/home/jerryant/Desktop/VICON-VIDEO-COMPARISON/"+"/"+participant+"/"+participant+"-graphs"+"/"

for root, dirs, files in os.walk(path+"/"+participant):
    for f in files:
        if '.png' in f:
            sep_path = root.split(os.sep)
            print sep_path
            print root
            copyfile(root+"/"+f, copy_to_path+"/"+sep_path[-2]+"-"+sep_path[-1]+"-"+f)