import tensorflow as tf
import numpy as np
import ReadBatch_utils as read_utils
import utils

LENTH_MAX = 32
LOAD_SIZE = 500
VOCABULARY_SIZE = 10000
EMBEDING_SIZE = 128
HIDEN_SIZE = 256
BATCH_SIZE = 10
ATTENTION_SIZE = 50
KEEP_PROB = 0.7
LEARNING_RATE = 0.01
CLASS_NUM = 5
MAX_INTERATION = 1000


def interface(input, keep_prob):
    with tf.variable_scope("Layer_1"):
        W1 = utils.weight_variable([5, 5, 1, 32], name="W1")
        b1 = utils.bias_variable([32], name="b1")
        conv_1 = utils.conv2d_basic(input, W1, b1)
        relu_1 = tf.nn.relu(conv_1, name="relu_1")
        ## pool_1: size [BATCH_SIZE, 32, 64, 32]
        pool_1 = utils.avg_pool_2x2(relu_1)

    with tf.variable_scope("Layer_2"):
        W2 = utils.weight_variable([5, 5, 32, 256], name="W2")
        b2 = utils.bias_variable([256], name="b2")
        conv_2 = utils.conv2d_basic(pool_1, W2, b2)
        relu_2 = tf.nn.relu(conv_2, name='relu_2')
        ## pool_2: size [BATCH_SIZE, 16, 32, 256]
        pool_2 = utils.avg_pool_2x2(relu_2)

    with tf.variable_scope("Layer_3"):
        W3 = utils.weight_variable([3, 3, 256, 384], name="W3")
        b3 = utils.bias_variable([384], name="b3")
        conv_3 = utils.conv2d_basic(pool_2, W3, b3)
        relu_3 = tf.nn.relu(conv_3, name="relu_3")
        ## pool_3: size [BATCH_SIZE, 8,16,384]
        pool_3 = utils.avg_pool_2x2(relu_3)

    with tf.variable_scope("Layer_4"):
        W4 = utils.weight_variable([3, 3, 384, 384], name="W4")
        b4 = utils.bias_variable([384], name="b4")
        conv_4 = utils.conv2d_basic(pool_3, W4, b4)
        relu_4 = tf.nn.relu(conv_4, name="relu_4")
        ## pool_4: size [BATCH_SIZE, 4,8, 384]
        pool_4 = utils.avg_pool_2x2(relu_4)

    with tf.variable_scope("Layer_5"):
        W5 = utils.weight_variable([3, 3, 384, 256], name="W5")
        b5 = utils.weight_variable([256], name="b5")
        conv_5 = utils.conv2d_basic(pool_4, W5, b5)
        relu_5 = tf.nn.relu(conv_5, name="relu_5")
        ## pool_5: size [BATCH_SIZE, 2, 4, 256]
        pool_5 = utils.avg_pool_2x2(relu_5)

    with tf.variable_scope("all_link"):
        W_fc1 = utils.weight_variable([1 * 4 * 256, 4096], name="W_fc1")
        b_fc1 = utils.bias_variable([4096], name="b_fc1")
        pool_5_flag = tf.reshape(pool_5, [-1, 1 * 4 * 256])
        ## h_fc1: size [BATCH_SIZE, 4096]
        h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(pool_5_flag, W_fc1, b_fc1))
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = utils.weight_variable([4096, 5], name="W_fc2")
        b_fc2 = utils.bias_variable([5], name="b_fc2")

        h_fc2 = tf.nn.tanh(tf.nn.xw_plus_b(h_fc1_drop, W_fc2, b_fc2))
        # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        return h_fc2


text_input = tf.placeholder(
    dtype=tf.int32, shape=[BATCH_SIZE, LENTH_MAX])
text_input_change = tf.reshape(text_input, [BATCH_SIZE, -1], name="Input")
keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
label = tf.placeholder(dtype=tf.float32, shape=[None, CLASS_NUM], name="label")

embeding_var = tf.Variable(tf.random_uniform(
    shape=[VOCABULARY_SIZE, EMBEDING_SIZE]), dtype=tf.float32, name='embeding_var')

## batch_embeding: size [BATCH_SIZE, LENTH_MAX, EMBEDING_SIZE]
batch_embeding = tf.nn.embedding_lookup(embeding_var, text_input_change)

batch_embeding_normal = tf.reshape(
    batch_embeding, [-1, LENTH_MAX, EMBEDING_SIZE, 1])

output = interface(batch_embeding_normal, keep_prob)
print(output)
print(label)

with tf.variable_scope("loss"):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    	logits=output, labels=label, name="loss"))


with tf.variable_scope("accuracy"):
# 	accuracy = tf.reduce_mean(
#   	  tf.cast(tf.equal(tf.round(tf.sigmoid(output)), label), dtype=tf.float32))
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.sigmoid(output),1), tf.argmax(label,1)), dtype=tf.float32))

train_op = tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(loss)

print("output:")
print(output)
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.image("language", )
summary_op = tf.summary.merge_all()

saver = tf.train.Saver()
accuracy_average = 0
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    train_datareader = read_utils.ReadBatch(
        LENTH_MAX, model="train", load_size=LOAD_SIZE, vocabulary_size=VOCABULARY_SIZE)
    test_datareader = read_utils.ReadBatch(
        LENTH_MAX, model="test", load_size=LOAD_SIZE, vocabulary_size=VOCABULARY_SIZE)

    summary_writer = tf.summary.FileWriter("./logs", sess.graph)

    for itr in range(MAX_INTERATION):
        x, y = train_datareader.next_text(BATCH_SIZE)
        feed_dict = {text_input: x, label: y, keep_prob: KEEP_PROB}
        sess.run(train_op, feed_dict)

        if itr % 10 == 0:
            loss_train, accuracy_train, summary_str = sess.run(
                [loss, accuracy, summary_op], feed_dict)

            print("Step: %d , Train_loss: %g ,Train_accuracy: %g" %
                  (itr, loss_train, accuracy_train))
            print(x[0])
            accuracy_average += accuracy_train
            summary_writer.add_summary(summary_str, itr)

        if itr % 100 == 0:
            x_test, y_test = test_datareader.next_text(BATCH_SIZE)
            loss_test, accuracy_test = sess.run([loss, accuracy], feed_dict={
                                                text_input: x_test, label: y_test, keep_prob: 1})
            print("Step: %d , loss_loss: %g ,loss_accuracy: %g" %
                  (itr, loss_test, accuracy_test))

    saver.save(sess.graph, "./log/model.ckpt")
num = MAX_INTERATION/10
accuracy_average = accuracy_average/num
print("accuracy_average: %g "%accuracy_average)