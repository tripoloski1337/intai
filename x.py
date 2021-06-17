import logParser
import pyinotify
import subprocess
FILE = "/var/log/nginx/access.log"

def readLast(p):
    # python implementation, but will run slow
    # with open(p + "access.log", "r") as file:
    #     for lastly in file:
    #         pass 
    lastly = subprocess.check_output(['tail', '-1', FILE])
    print(lastly)


class ModHandler(pyinotify.ProcessEvent):
    # evt has useful properties, including pathname
    def process_IN_MODIFY(self, evt):
        readLast("/var/log/nginx/")


handler = ModHandler()
wm = pyinotify.WatchManager()
notifier = pyinotify.Notifier(wm, handler)
wdd = wm.add_watch(FILE, pyinotify.IN_MODIFY)
notifier.loop()