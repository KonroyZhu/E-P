import numpy as np
import tensorflow as tf

from com.ensemble_record import esm_record
from com.preprocess import transform_data_to_id, seg_data
from com.utils import pad_answer, padding
import argparse
import pickle
import codecs



parser = argparse.ArgumentParser(description='inference procedure, note you should train the data at first')

parser.add_argument('--data', type=str,
                    default='data/ai_challenger_oqmrc_testa.json',
                    help='location of the test data')

parser.add_argument('--word_path', type=str, default='data/word2id.obj',
                    help='location of the word2id.obj')

parser.add_argument('--output', type=str, default='prediction.a.txt',
                    help='prediction path')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='batch size')

args = parser.parse_args()

def pad_wrong_answer(answer_list):
    # 3680
    # 7136
    # 这两批数据中有alternative answer长度小于3的数据，需要补齐否则无法处理
    # 该方法通过复制ans[0]补齐数据
    padded_list=[]
    for ans in answer_list:
        ll=len(ans)
        if not ll == 3:
            for _ in range(3-ll):
                ans+=[ans[0]]
        padded_list.append(ans)
    padded_list=pad_answer(padded_list,70)
    return padded_list


def inference(pred,query,para,ans,sess,data,store_path="esm_record/test.pkl"):
    # model.eval()
    predictions = []
    exception=[]
    id_list = []  # 用于记录ensemble所需的数据
    pred_list = []
    # with torch.no_grad():
    for i in range(0, len(data), args.batch_size):
        one = data[i:i + args.batch_size]
        q, _ = padding([x[0] for x in one], max_len=50)
        p, _ = padding([x[1] for x in one], max_len=350)
        a = pad_answer([x[2] for x in one],max_length=70)
        str_words = [x[-1] for x in one]
        ids = [x[3] for x in one]
        if not len(np.shape(a)) == 3:
            # print(i)
            a = pad_wrong_answer(a)
        # query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
        # if args.cuda:
        #     query = query.cuda()
        #     passage = passage.cuda()
        #     answer = answer.cuda()
        output =np.argmax(sess.run(pred,feed_dict={
            query:q,
            para:p,
            ans:a
        }),axis=1)
        # id_list = id_list.extend(ids)
        # pred_list = pred_list.extend(output)
        output = list(output)
        ids = list(ids)
        id_list.extend(ids)
        pred_list.extend(output)
        for q_id, prediction, candidates in zip(ids, output, str_words):
            print(q_id)
            print(prediction)
            #FIXME
            l=len(candidates)
            fir=candidates[0]
            if l<3:
                for _ in range(3-l):
                    candidates.append(fir)
            print(candidates)
            prediction_answer = u''.join(candidates[prediction])
            predictions.append(str(q_id) + '\t' + prediction_answer)
    outputs = u'\n'.join(predictions)
    with codecs.open(args.output, 'w',encoding='utf-8') as f:
        f.write(outputs)
    print ('done!')

    esm_record(id_list=id_list, pred_list=pred_list, path=store_path)

def load_data(w2id_path="data/word2id.obj",seg_path="data/test_seg.pkl"):
    print("data loading...")
    with open(w2id_path, 'rb') as f:
        word2id = pickle.load(f)
    raw_data = pickle.load(open(seg_path, "rb"))[:100]
    transformed_data = transform_data_to_id(raw_data, word2id)
    data = [x + [y[2]] for x, y in zip(transformed_data, raw_data)]
    data = sorted(data, key=lambda x: len(x[1]))
    print('test data size {:d}'.format(len(data)))
    return data

def load_net(session, path="net/rnet/model.ckpt"):
    """
    根据路径加载模型
    :param session:
    :param path:
    :return:
    """
    saver = tf.train.import_meta_graph(path + ".meta")
    print("restore...")
    saver.restore(session, path)  # 到.ckpt即可，saver中的命名一致
    graph = tf.get_default_graph()
    print("pred restore...")
    pred = graph.get_tensor_by_name("Prediction_Layer/score:0")  # 取 tensor
    query = graph.get_tensor_by_name("query:0")  # 取 tensor
    para = graph.get_tensor_by_name("para:0")  # 取 tensor
    ans = graph.get_tensor_by_name("ans:0")  # 取 tensor

    return pred, query, para, ans

if __name__ == '__main__':
    test_dt=load_data(args.word_path,seg_path="data/test_seg.pkl")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pred, query, para, ans=load_net("net/rnet/model.ckpt")
        inference(pred,query,para,ans,sess,test_dt,store_path="esm_record/test.pkl")
# """

