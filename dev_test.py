import numpy as np
import tensorflow as tf
import pickle

from com.utils import padding, pad_answer


def test(pred,session,ans,que,para):
    r, a = 0.0, 0.0
    id_list = []  # 用于记录ensemble所需的数据
    pred_list = []
    for i in range(0, len(dev_data),10):
        one = dev_data[i:i +10]
        query, _ = padding([x[0] for x in one], max_len=350)
        passage, _ = padding([x[1] for x in one], max_len=50)
        answer = pad_answer([x[2] for x in one],max_len=70)
        ids = [int(c[3]) for c in one]
        # query, passage, answer = np.array(query), np.array(passage), np.array(answer)
        fd = {
            para: passage,
            que: query,
            ans: answer
        }
        p = session.run(pred, feed_dict=fd)

        # 储存q_id与预测答案下标
        p = list(p)
        ids = list(ids)
        id_list.extend(ids)
        pred_list.extend(p)

        r=0
        for item in p:
            if np.argmax(item) == 0:
                r+=1
        a += len(one)
    return r * 100.0 / a ,id_list,pred_list

print("loading data...")
with open('data/dev.pickle', 'rb') as f:
    dev_data = pickle.load(f)
dev_data = sorted(dev_data, key=lambda x: len(x[1]))
print("loding session...")
saver=tf.train.import_meta_graph("net/model.ckpt.meta")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "net/model.ckpt") # 到.ckpt即可，saver中的命名一致
    graph = tf.get_default_graph()
    print("restoring...")
    pred = graph.get_tensor_by_name("pred:0") # 取 tensor
    query = graph.get_tensor_by_name("query:0")  # 取 tensor
    para = graph.get_tensor_by_name("para:0")  # 取 tensor
    ans = graph.get_tensor_by_name("ans:0")  # 取 tensor
    print("testing...")
    acc,_,_=test(pred,sess,ans,query,para)
    print(acc)
    
