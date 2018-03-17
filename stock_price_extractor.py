import requests
import base64
import csv

username = "d337f69c0a8b4db5841d08ed55313214"
password = "dadd178bdf8655e543c4b983a9bf22f0"
b64Val = base64.b64encode(b"d337f69c0a8b4db5841d08ed55313214:dadd178bdf8655e543c4b983a9bf22f0").decode("ascii")
ticker = "AAPL"

values = []
for page in range(1, 11):

    r = requests.get("https://api.intrinio.com/prices?identifier=" + ticker  + "&page_number=" + str(page),
                    headers={"Authorization": "Basic %s" % b64Val})
    
    for row in r.json()["data"]:
        
        dicts = {}
        dicts["Date"] = row["date"]
        dicts["Opening Price"] = row["open"]
        dicts["Closing Price"] = row["close"]
        dicts["Highest Price"] = row["high"]
        dicts["Volume"] = row["volume"]
        values.append(dicts)  


fields = ['Date', 'Opening Price', 'Closing Price', 'Highest Price', 'Volume']
filename = "stock_prices_" + ticker + ".csv"


with open(filename, 'w', newline='') as csvfile:
    
    writer = csv.DictWriter(csvfile, fieldnames = fields)

    writer.writeheader()
    
    writer.writerows(values)

