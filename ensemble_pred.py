import numpy as np
import pickle
import codecs

from ensemble import swap_dt
from single_pred import load_data

if __name__ == '__main__':
    data=load_data(w2id_path="data/word2id.obj",seg_path="data/test_seg.pkl")
    rnet_dev=pickle.load(open("esm_record/rnet/train.pkl","rb")) # FIXME should be test.pkl here
    mwan_dev=pickle.load(open("esm_record/mwan/train.pkl","rb"))
    esm_model=pickle.load(open("models/ada_rnet_mwan.pkl","rb"))
    x=[]
    y=[]
    ids=[]
    str_words=[]
    data_dict={}
    for dt in data:
        data_dict[dt[3]]=dt

    for k in rnet_dev.keys():
        # str_words.append(data_dict[k][-1]) Fixme：当train.pkl换成test.pkl的时候才能够取str_words(train中没有test的q_id)
        x.append([rnet_dev[k],mwan_dev[k]])
        ids.append(k)
    pred=esm_model.predict(x)

#################################
    output_path="some.txt"
    predictions = []
    for q_id, prediction, candidates in zip(ids, pred, str_words):
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
    with codecs.open(output_path, 'w',encoding='utf-8') as f:
        f.write(outputs)

