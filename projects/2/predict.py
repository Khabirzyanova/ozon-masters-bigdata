#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')
from model import valid_fields

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("2.joblib")

#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

#read and infere
read_opts=dict(
        sep='\t', names=[fields[0]]+fields[2:], index_col='id',
        iterator=True, chunksize=100000, na_values='\\N'
)
rec_counter = 0
for df in pd.read_csv(sys.stdin, **read_opts):
    if len(df) == 0:
        logging.info(f"found 0 length dataframe")
        continue
    df['day_number'] = "day_0"
    pred = model.predict_proba(df)
    out = zip(df.index, pred[:, 0])
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
    rec_counter += len(df)
    logging.info(f"processed {rec_counter}")
