import pandas as pd
import sqlite3
conn = sqlite3.connect('INSTRUCTOR.db')
cursor_obj = conn.cursor()
cursor_obj.execute("DROP TABLE IF EXISTS INSTRUCTOR")
table = """ create table IF NOT EXISTS INSTRUCTOR(
    ID INTEGER PRIMARY KEY NOT NULL, 
    FNAME VARCHAR(20), 
    LNAME VARCHAR(20), 
    CITY VARCHAR(20), 
    CCODE CHAR(2));"""
cursor_obj.execute(table)
# print('Table is Ready')
cursor_obj.execute('''ALTER TABLE INSTRUCTOR ADD Country varchar(20)''')
cursor_obj.execute(''' insert into INSTRUCTOR values(1,'Rav','Ahuja','TORONTO','CA','Canada'),
                   (2, 'Raul', 'Chong', 'Markham', 'CA','Canada'),
                    (3, 'Hima', 'Vasudevan', 'Chicago', 'US','United State'),
                   (4,"Muhammad Umer","Khan","Karachi","PK","Pakistan") ''')

# statment = '''select * from INSTRUCTOR'''
# cursor_obj.execute(statment)
# print('All the Data')

# output_all = cursor_obj.fetchall()
# for row_all in output_all:
#     print(row_all)

## If you want to fetch few rows from the table we use fetchmany(numberofrows) and mention the number how many rows you want to fetch
# output_many = cursor_obj.fetchmany(2)
# for row_many in output_many:
#     print(row_many)

## Fetch only FNAME from the table
# statment = '''select FNAME from INSTRUCTOR'''
# cursor_obj.execute(statment)
# print('All the Data')
# output_fetch = cursor_obj.fetchall()
# output_fetch = cursor_obj.fetchmany(1)
# for fetch in output_fetch:
#     print(fetch)


## UPDATING DATA --- Changing city to MOOSETOWN where FNAME = 'Rav'
# query_update = '''update INSTRUCTOR set CITY = "MOOSETOWN" WHERE FNAME = "Rav" '''
# cursor_obj.execute(query_update)
# statement = '''select * from INSTRUCTOR'''
# cursor_obj.execute(statement)
# print('All Data')
# output_update = cursor_obj.fetchall()
# for row_update in output_update:
#     print(row_update)

## Retrieve data into Pandas
df = pd.read_sql_query("select * from INSTRUCTOR",conn)
# print(df)
# print(df.shape) # Output (3,5) 3 columns, and 5 rows

# Creating another table
cursor_obj.execute("DROP TABLE IF EXISTS EMPLOYEES")
table1=('''create table EMPLOYEES(
            EMPLOYEE_ID PRIMARY KEY NOT NULL,
            FNAME VARCHAR(10),
            LNAME VARCHAR(10),
            SSN INT,
            B_DATE varchar(15),SEX varchar(1),ADDRESS varchar(50), JOB_ID INT,SALARY INT, MANAGER_ID INT, DEP_ID,DEP_NAME
) ''')
print('\n\n------EMPLOYEES------\n')
cursor_obj.execute(table1)
cursor_obj.execute('''insert into EMPLOYEES values('E1001','John','Thomas','123456','1976-01-09','M','5631 Rice, OakPark,JL',100,1000000,30001,2,'Architect Group')
                   ,('E1002', 'Alicw', 'James', '123457', '1972-07-31', 'F', '980 Berry In,Elgin,JL', '200', '80000', '30002', '5', 'Software Development'),
                    ('E1003', 'Steve', 'Wells', '123458', '1980-08-10', 'M', '291 Springs,Gary,JL', '300', '50000', '30003', '5', 'Design Team')
                   ''')
df1=pd.read_sql_query("select * from EMPLOYEES",conn)
# print(df1)


conn.close()