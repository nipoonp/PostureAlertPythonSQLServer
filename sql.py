# import pyodbc



# con = pyodbc.connect("DRIVER={SQL Server};server=localhost;database=TestDB1;Trusted_Connection=yes")
# cur = con.cursor()
# # cur.execute("select * from celebs")


import pymysql

connection = pymysql.connect(host='localhost', user='Nipoon', password='890xyz', db='celebs')
cursor=connection.cursor()
sql=("SELECT * FROM celebs")
cursor.execute(sql)
data=cursor.fetchall()
print(data)