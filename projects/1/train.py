
#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

#
# Import model definition
#
from model import model, fields


#
# Logging initialization
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#
# Read script arguments
#
try:
  proj_id = sys.argv[1] 
  train_path = sys.argv[2]
except:
  logging.critical("Need to pass both project_id and train dataset path")
  sys.exit(1)


logging.info(f"TRAIN_ID {proj_id}")
logging.info(f"TRAIN_PATH {train_path}")

#
# Read dataset
#
#fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
#num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)

#split train/test
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:,:-1], df.iloc[:,-1], test_size=0.33, random_state=42
)

# TRAIN_SPLIT = 0.2
# VALIDATION_SPLIT = 0.5

# ds = tf.data.Dataset.zip((
#     tf.data.Dataset.from_tensor_slices((
#         tf.cast(df[dense_cols].values, tf.float32),
#         tf.cast(df[cat_cols].values, tf.int32),
#     )),
#     tf.data.Dataset.from_tensor_slices((
#         tf.cast(to_categorical(df['label'].values, num_classes=2), tf.float32)
#     ))
# )).shuffle(buffer_size=2048)


# ds_test = ds.take(int(len(ds) * TRAIN_SPLIT))
# ds_train = ds.skip(len(ds_test))
# ds_valid = ds_test.take(int(len(ds_test) * VALIDATION_SPLIT))
# ds_test = ds_test.skip(len(ds_valid))



#
# Train the model

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#     metrics=['accuracy']
# )

# BATCH_SIZE = 128

# model.fit(
#   ds_train.batch(BATCH_SIZE),
#   validation_data=ds_valid.batch(BATCH_SIZE),
#   callbacks=[
#     tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
#     ],
#   epochs=100,
#   verbose=1,
# )

# model_score = model.evaluate(ds_test.batch(BATCH_SIZE))
# # print(f'Loss {results[0]}, Accuracy {results[1]}')
# logging.info(f"model score: {model_score[0]:.3f}")


#
model.fit(X_train, y_train)

model_score = model.score(X_test, y_test)

logging.info(f"model score: {model_score:.3f}")

# save the model
dump(model, "{}.joblib".format(proj_id))


# # needed part according to hw 
# dump(model, "1.joblib")
