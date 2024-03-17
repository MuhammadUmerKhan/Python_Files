from practice_sql import *
read=pd.read_sql_query('select * from JOB',conn)
print(read)