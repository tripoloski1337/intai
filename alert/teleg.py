import requests
from util.consolelog import Console


class Teleg:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.msg = ''
        self.url = ""
    
    def setReporter(self, ip, status, attack_vector, url):
        self.msg = '''
[Intai Reporter]
Your web server is under attack
We found malicious activity on your web server!!
IP: {}
http status: {}
Attack vector: {}
url req: {}
'''.format(ip, status, attack_vector, url)
    def fire(self):
        self.url = "https://api.telegram.org/bot" + self.token + "/sendMessage?chat_id=" + self.chat_id + "&text=" + self.msg
        x = requests.get(self.url)
        res = x.json()
        if res['ok'] == True:
            Console.info("msg sent")
        else:
            Console.danger("msg fail to sent")

