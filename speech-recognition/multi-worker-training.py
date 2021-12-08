import json
import os

import tensorflow as tf

import base

per_worker_batch_size = 32
tf_config = json.loads(os.environ["TF_CONFIG"])
num_workers = len(tf_config["cluster"]["worker"])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset, num_to_char, char_to_num = base.ctc_dataset(global_batch_size)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = base.build_and_compile_ctc_model(num_to_char, char_to_num)


multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
