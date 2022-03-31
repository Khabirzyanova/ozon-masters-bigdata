insert overwrite directory 'Khabirzyanova_hiveout'
row format delimited
fields terminated by '\t' 
stored as textfile select * from hw2_pred; 
