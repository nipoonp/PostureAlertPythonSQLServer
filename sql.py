# import pyodbc



# con = pyodbc.connect("DRIVER={SQL Server};server=localhost;database=TestDB1;Trusted_Connection=yes")
# cur = con.cursor()
# # cur.execute("select * from celebs")


import pymysql

connection = pymysql.connect(host='localhost', user='root', password='12345678', db='PostureAlert')
cursor=connection.cursor()
sql=("SELECT * FROM SensorReadings WHERE Posture IS NULL")
cursor.execute(sql)
data=cursor.fetchall()
print(data)