from termcolor import colored

class Console:
    def px(dt, ip, method, bytessent, referrer, ua, stat, url):
        print("[{}] [{}] {} {} {}".format(colored(dt, "blue"), colored(method,"yellow"), colored(ip, "green"), colored(stat, "cyan"), colored(url,"red")))
        # print(colored("[" + dt + "]", "blue") + " " + colored("[" + method + "]", "yellow") + " " + colored(ip, "green")) + " " + stat + " " + colored(url, "red")

    def info(msg):
        print("{} {}".format(colored("[?]", "blue"), colored(msg, "green")))
    
    def warning(msg):
        print("{} {}".format(colored("[!]", "yellow"), colored(msg, "green")))
    
    def danger():
        print("{} {}".format(colored("[!]", "red"), colored(msg, "green")))