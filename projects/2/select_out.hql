insert overwrite directory 'Khabirzyanova_hiveout'
row format delimited
fields terminated by '\t' 
stored as textfile select * from Khabirzyanova.hw2_pred; 
