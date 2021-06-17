#!/usr/bin/env python
def readmodel(path):
    raw = open(path).read()
    blacklist = raw.split()
    return blacklist
