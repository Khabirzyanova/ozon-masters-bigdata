CREATE TABLE hw2_pred (
  id int,
  pred float)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION 'Khabirzyanova_hw2_pred';
