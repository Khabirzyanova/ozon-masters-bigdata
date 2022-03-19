insert overwrite directory 'Khabirzyanova_hiveout'
row format delimited
fields terminated by '\t' 
select id, pred from hw2_pred; 
