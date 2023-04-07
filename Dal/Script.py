import requests
import time
import json

url1 = "https://api.thingspeak.com/channels/1785779/feeds/last.json?api_key=64LX24ENHBDW3JZV"
url2 = "https://api.thingspeak.com/channels/1780792/feeds/last.json?api_key=6PKCOVRLO9U2HFJ7"
url3 = "https://api.thingspeak.com/channels/1744711/feeds/last.json?api_key=LQYCBT76OZKU05HV"
url4 = "https://api.thingspeak.com/channels/1744701/feeds/last.json?api_key=2HY016JWRK0O7IFI"

while True:
    response1 = requests.get(url1)
    data_disc1 = json.loads(response1.text)
    input1 =  data_disc1['field1']

    response2 = requests.get(url2)
    data_disc2 = json.loads(response2.text)
    input2 =  data_disc2['field1']

    response3 = requests.get(url3)
    data_disc3 = json.loads(response3.text)
    input3 = data_disc3['field1']

    response4 = requests.get(url4)
    data_disc4 = json.loads(response4.text)
    input4 = data_disc4['field1']

    time.sleep(30)



