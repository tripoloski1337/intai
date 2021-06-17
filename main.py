import logParser
import pyinotify
import modelReader
import subprocess
import configparser
from util.consolelog import Console
config = configparser.ConfigParser()
config.read("./config/config.ini")
FILE = config['LOGGER']['nginx_log']

def welcome():
    x = '''                            
        o         o          o 
        8         8            
        8 odYo.  o8P .oPYo. o8 
        8 8' `8   8  .oooo8  8 
        8 8   8   8  8    8  8 
        8 8   8   8  `YooP8  8  an Intrusion detection system with neural network algorithm
        ....::..::..::.....::.. author: Muhammad Arsalan Diponegoro
        ::::::::::::::::::::::: Universitas Gunadarma
        :::::::::::::::::::::::
'''
    print(x)

def readLast():
    # python implementation, but will run slow
    # with open(p + "access.log", "r") as file:
    #     for lastly in file:
    #         pass 
    lastly = subprocess.check_output(['tail', '-1', FILE])
    logParser.parse((str(lastly).replace("\n","")))


class ModHandler(pyinotify.ProcessEvent):
    # evt has useful properties, including pathname
    def process_IN_MODIFY(self, evt):
        readLast()

def main():
    welcome()
    Console.info("preparing..")
    handler = ModHandler()
    wm = pyinotify.WatchManager()
    notifier = pyinotify.Notifier(wm, handler)
    wdd = wm.add_watch(FILE, pyinotify.IN_MODIFY)
    notifier.loop()

if __name__ == "__main__":
    main()
