# For resore test
import tensorflow as tf

saver = tf.train.import_meta_graph("tmp/tfmodel/model.ckpt-0.meta")
sess = tf.Session()
saver.restore(sess, r"tmp/tfmodel/model.ckpt-0")
saver.save(sess, "tmp/model.ckpt")
