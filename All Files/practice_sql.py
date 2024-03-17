import pandas as pd
import sqlite3
conn = sqlite3.connect('INSTRUCTOR.db')
cursor_obj=conn.cursor()
cursor_obj.execute("DROP TABLE IF EXISTS EMPLOYEES")
table1=('''create table EMPLOYEES(
            EMPLOYEE_ID PRIMARY KEY NOT NULL,
            FNAME VARCHAR(10),
            LNAME VARCHAR(10),
            SSN INT,
            B_DATE varchar(15),SEX varchar(1),ADDRESS varchar(50), JOB_ID INT,SALARY INT, MANAGER_ID INT, DEP_ID,DEP_NAME
) ''')
# print('\n\n------EMPLOYEES------\n')
cursor_obj.execute(table1)
cursor_obj.execute('''insert into EMPLOYEES values('E1001','John','Thomas','123456','1976-01-09','M','5631 Rice, OakPark,JL',100,1000000,30001,2,'Architect Group')
                   ,('E1002', 'Alicw', 'James', '123457', '1972-07-31', 'F', '980 Berry In,Elgin,JL', '200', '80000', '30002', '5', 'Software Development'),
                    ('E1003', 'Steve', 'Wells', '123458', '1980-08-10', 'M', '291 Springs,Gary,JL', '300', '50000', '30003', '5', 'Design Team')
                   ''')
df1=pd.read_sql_query("select * from EMPLOYEES",conn)
# print(df1)
cursor_obj.execute('DROP TABLE IF EXISTS DEPARTMENT')
table2=('''create table IF NOT EXISTS DEPARTMENT(
        DEPT_ID_DEP int,
        DEP_NAME VARCHAR(20),
        MANAGER_ID INT,
        LOC_ID VARCHAR(7) 
)''')
cursor_obj.execute(table2)
cursor_obj.execute('''
        insert into DEPARTMENT values(2, 'Architect Group', '30001', 'L0001'),
                   (5, 'Software Developer', '30002', 'L0002'),
                   (7, 'Design Team', '3003', 'L0003'),
                   (7, 'Design Team', '3003', 'L0003')
''')
df2=pd.read_sql_query("select * from DEPARTMENT",conn)
# print(df2)

cursor_obj.execute('DROP TABLE IF EXISTS JOB_HISTORY')
table3=('''create table IF NOT EXISTS JOB_HISTORY(
        EMPLOYEE_ID VARCHAR(10) PRIMARY KEY NOT NULL,
        START_DATE VARCHAR(20),
        JOBS_ID INT,    
        DEPT_ID INT 
)''')
cursor_obj.execute(table3)
cursor_obj.execute('''insert into JOB_HISTORY values('E1001', '2000-01-30', 100, 2),
                   ('E1002', '2000-01-30', 200, 5),
                   ('E1003', '2000-01-30', 300, 5)''')
df3=pd.read_sql_query("select * from JOB_HISTORY",conn)
# print(df3)
cursor_obj.execute('DROP TABLE IF EXISTS JOB')
table4=('''create table IF NOT EXISTS JOB(
                   JOB_IDENT int,
                   JOB_TITLE VARCHAR(20),
                    MIN_SALARY INT,
                    MAX_SALARY INT

)''')
cursor_obj.execute(table4)
cursor_obj.execute('''insert into JOB values
                         ('100','Sr.Architect','60000','100000'),
                         ('200','Sr.SoftwareDeveloper','60000','80000'),
                         ('300','Jr.SoftwareDeveloper','40000','60000')
''')
df4=pd.read_sql_query("select * from JOB",conn)
# print(df4)

cursor_obj.execute('DROP TABLE IF EXISTS LOCATIONS')
table5=('''create table IF NOT EXISTS LOCATIONS(
        LOC_ID varchar(5),
	DEP_ID_LOC int
)''')
cursor_obj.execute(table5)
cursor_obj.execute('''insert into LOCATIONS values
        ('L0001',2),
        ('L0002',5),
        ('L0003',7);
''')
df5=pd.read_sql_query("select * from LOCATIONS",conn)
print(df5)

# conn.close()