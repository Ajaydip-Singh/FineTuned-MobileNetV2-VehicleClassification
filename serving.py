import tensorflow as tf
from ModelDetector import *

# create a model
detector = ModelDetector()
detector.model.load_weights("./trained/model_detector.h5")

model = detector.model

prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
    inputs={'x':  model.input},
    outputs={'y': model.output}
)

with tf.keras.backend.get_session() as sess:
    builder = tf.saved_model.builder.SavedModelBuilder('./trained/model_detector/1/')
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                             tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                 prediction_signature})
    builder.save()

print('Export complete.')

# execute this line in terminal after installing docker
# WORKSPACE='write your workspace using absolute path'
# docker run -d -p 81:8501 -v $WORKSPACE/trained:/models -e MODEL_NAME=model_detector -e MODEL_BASE_PATH=/models -t tensorflow/serving:1.14.0
