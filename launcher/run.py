class Runner:
    def readLast(p):
        # python implementation, but will run slow
        # with open(p + "access.log", "r") as file:
        #     for lastly in file:
        #         pass 
        lastly = subprocess.check_output(['tail', '-1', FILE])
        logParser.parse((str(lastly).replace("\n","")))

