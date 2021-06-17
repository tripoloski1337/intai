#!/usr/bin/env python
import gzip
import os
import sys
import re
from util.consolelog import Console
from core.train import MachineLearning
from core.detector import Detector
from time import time
from alert.teleg import Teleg
import configparser

config = configparser.ConfigParser()
config.read("./config/config.ini")


# INPUT_DIR = "nginx-logs"
# def px(dt, ip, method, bytessent, referrer, ua, stat, url):
#     print("[{}] [{}] {} {} {}".format(colored(dt, "blue"), colored(method,"yellow"), colored(ip, "green"), colored(stat, "cyan"), colored(url,"red")))
#     # print(colored("[" + dt + "]", "blue") + " " + colored("[" + method + "]", "yellow") + " " + colored(ip, "green")) + " " + stat + " " + colored(url, "red")

# ML = MachineLearning()
# ML.preps_predict("./core/dataset/sqli.csv", "./core/model/sqli.h5")

sqli_detect = Detector("./core/dataset/sqli.csv", "./core/model/sqli.h5", "utf-16")
sqli_detect.ignite()
xss_detect = Detector("./core/dataset/xss.csv","./core/model/xss.h5", "utf-8")
xss_detect.ignite()

tele = Teleg(config['TELEGRAM']['token'], config['TELEGRAM']['chat_id'])

def parse(INPUT_DIR):
    lineformat = re.compile(r"""(?P<ipaddress>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) - - \[(?P<dateandtime>\d{2}\/[a-z]{3}\/\d{4}:\d{2}:\d{2}:\d{2} (\+|\-)\d{4})\] ((\"(GET|POST) )(?P<url>.+)(http\/1\.1")) (?P<statuscode>\d{3}) (?P<bytessent>\d+) (?P<refferer>-|"([^"]+)") (["](?P<useragent>[^"]+)["])""", re.IGNORECASE)
    x = []
    data = re.search(lineformat, INPUT_DIR)
    if data:
        datadict = data.groupdict()
        ip = datadict["ipaddress"]
        datetimestring = datadict["dateandtime"]
        url = datadict["url"]
        bytessent = datadict["bytessent"]
        referrer = datadict["refferer"]
        useragent = datadict["useragent"]
        status = datadict["statuscode"]
        method = data.group(6)

        x.append([ip, \
        datetimestring, \
        url, \
        bytessent, \
        referrer, \
        useragent, \
        status, \
        method])
        
        print("+--------------------------------------------+")
        Console.px(datetimestring, ip, method, bytessent, referrer, useragent, status, url)
        attack_vector = ""
        sqli_pred = sqli_detect.check(url[1:])
        Console.info("SQLI prediction: " + str(sqli_pred))
        xss_pred = xss_detect.check(url)
        Console.info("XSS prediction: " + str(xss_pred))
        if((sqli_pred) == 1):
            attack_vector += "SQL Injection, "
            Console.warning("SQL Injection attempt")
        if((xss_pred) == 1):
            attack_vector += "XSS Cross-site scripting, "
            Console.warning("XSS Cross-site scripting")

        if(config['TELEGRAM']['enable_telegram'] == "1"):
            # Prepare to send if there's any misc activity
            if xss_pred == 1 or sqli_pred == 1:
                tele.setReporter(ip, status, attack_vector, url)
                tele.fire()
        print("+--------------------------------------------+")
            

        
    return x
    logfile.close()
