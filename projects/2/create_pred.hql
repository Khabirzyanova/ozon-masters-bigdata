use Khabirzyanova;
create table hw2_pred(
  id int,
  pred float)
row format delimited
fields terminated by '\t'
stored as textfile
location 'Khabirzyanova_hw2_pred';
