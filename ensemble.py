import pickle

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

def scoring(model,_x,_y,time=5):
    ls=[]
    for i in range(time):
        print("{} in {}".format(i,time))
        pred = model.predict(_x)
        hit=0.0
        for i in range(len(pred)):
            if pred[i] == _y[i]:
                hit+=1
        ls.append(hit/len(pred))
    print(ls)
def net_record(record_path="esm_record",net="rnet",mode="dev"):
    if not record_path.endswith("/"):
        record_path+="/"
    if not net.endswith("/"):
        net+="/"
    path=record_path+net
    path+=mode+".pkl"
    pkl=pickle.load(open(path,"rb"))
    return pkl


def acc_on_dev(net="rnet",mode="dev"):
    """
    测试dev集上的准确率
    :param net:
    :return:
    """
    dev=net_record(net="rnet",mode=mode)
    hit=0.0
    for k in dev.keys():
        # print(k,dev[k])
        if dev[k] == 0:
            hit+=1
    print("model: {} hit: {} dev size: {}".format(net,hit/len(dev),len(dev)))

def swap_dt(data):
    """
    源数据集中正确下标全为0
    为不让集成器学习到全0的输出此处对训练集的下标作交换操作：
    0-1/3：不作变换
    1/3-2/3：下标0与下标1交换
    2/3-1：下标0与下标2交换
    :param data:
    :return: data_swap: {q_id:[pred,label]} # pred 为预测值，label 为正确答案
    """
    def swap(dt, a, b):
        if dt == a:
            return b
        if dt == b:
            return a
        else:
            return dt

    ll=len(data)
    data_swaped={}
    for i,k in enumerate(data.keys()):
        if i < ll/3:
            # print("0-1/3:",i)
            data_swaped[k] = [data[k],0] # 不作改变，仍然以0作正确下标
        elif i >= ll/3 and i<2*ll/3:
            # print("1/3-2/3:", i)
            data_swaped[k] = [swap(data[k], 0, 1),1]
        else:
            # print("2/3-1:",i)
            data_swaped[k] = [swap(data[k], 0, 2),2]
    return data_swaped
if __name__ == '__main__':
    train_r=net_record(net="rnet",mode="train")
    train_m=net_record(net="mwan",mode="train")
    dev_r=net_record(net="rnet",mode="dev")
    dev_m=net_record(net="mwan",mode="dev")
    acc_on_dev("rnet",mode="dev")
    acc_on_dev("mwan",mode="dev")
    rnet_train=swap_dt(train_r)
    mwan_train=swap_dt(train_m)
    rnet_dev=swap_dt(dev_r)
    mwan_dev=swap_dt(dev_m)


    X_train=[]
    y_train=[]
    train_keys=[k for k in rnet_train.keys()]
    np.random.shuffle(train_keys)
    for k in train_keys:
        assert rnet_train[k][1] == mwan_train[k][1]
        # print("pred r:{} , m: {}".format(rnet_x_y[k][0],mwan_x_y[k][0]),"  lab: {}".format(rnet_x_y[k][1]))
        X_train.append([rnet_train[k][0], mwan_train[k][0]])
        y_train.append(rnet_train[k][1])
    X_dev=[]
    y_dev=[]
    dev_keys=[k for k in rnet_dev.keys()]
    np.random.shuffle(dev_keys)
    for k in dev_r.keys():
        assert rnet_dev[k][1] == mwan_dev[k][1]
        X_dev.append([rnet_dev[k][0],mwan_dev[k][0]])
        y_dev.append(rnet_dev[k][1])
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_dev=np.array(X_dev)
    y_dev=np.array(y_dev)

    model=AdaBoostClassifier(learning_rate=0.1,n_estimators=20) #0.5722 #best: {'learning_rate': 0.1, 'algorithm': 'SAMME.R', 'n_estimators': 20} : 0.5741
    # model=GradientBoostingClassifier() #0.5741
    # model=RandomForestClassifier(n_estimators=100) # 0.5741
    # model=SVC(gamma="scale")
    # model=BernoulliNB() # 0.5348666666666667
    # model=MultinomialNB() # 0.4483
    print("training ...")
    model.fit(X_train,y_train)
    pickle.dump(model,open("models/ada_rnet_mwan.pkl","wb"))
    scoring(model,X_dev,y_dev)
    print("score:",model.score(X_dev,y_dev))


    # # grid
    # print("grid searching..")
    # from sklearn.model_selection import GridSearchCV, PredefinedSplit
    # parameters = { 'n_estimators':[10,40,70,100,130,160,200]
    #                }
    #
    # X_train_dev = np.concatenate((X_train, X_dev), axis=0)
    # y_train_dev = np.concatenate((y_train, y_dev), axis=0)
    # test_fold = np.zeros(X_train_dev.shape[0])   # 将所有index初始化为0,0表示第一轮的验证集
    # test_fold[:X_train.shape[0]] = -1            # 将训练集对应的index设为-1，表示永远不划分到验证集中
    # ps = PredefinedSplit(test_fold=test_fold)
    # grid_search_params = {'estimator': model,             # 目标分类器
    #                       'param_grid': parameters,  # 前面定义的我们想要优化的参数
    #                       'cv': ps,                     # 使用前面自定义的split验证策略
    #                       'n_jobs': -1,                 # 并行运行的任务数，-1表示使用所有CPU
    #                       }
    #
    # grsearch = GridSearchCV(**grid_search_params)
    # grsearch.fit(X_train_dev, y_train_dev)
    #
    # print("best:",grsearch.best_params_)
    # print("socre:",grsearch.best_score_)

    #############################################################################
    # from sklearn.model_selection import train_test_split, PredefinedSplit
    # from sklearn.metrics import make_scorer
    # def evaluate_on_dev(estimator, X, y,dev):
    #     X=dev["x"]
    #     y=dev["y"]
    #     p=estimator.predict(X)
    #     hit=0.0
    #     for i in range(len(y)):
    #         if p[i] == y[i]:
    #             hit+=1
    #     return hit/len(y)
    #
    #
    # import  numpy as np
    # from sklearn import svm, datasets
    # from sklearn.model_selection import GridSearchCV
    # iris = datasets.load_iris()
    # X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)
    #
    #
    # parameters = {'kernel':('sigmoid','linear'), 'C':[1, 10]}
    # model = svm.SVC(gamma="scale")
    # train_val_features = np.concatenate((X_train,X_test ),axis=0)
    # train_val_labels = np.concatenate((y_train,y_test ),axis=0)
    # test_fold = np.zeros(train_val_features.shape[0])   # 将所有index初始化为0,0表示第一轮的验证集
    # test_fold[:X_train.shape[0]] = -1            # 将训练集对应的index设为-1，表示永远不划分到验证集中
    # ps = PredefinedSplit(test_fold=test_fold)
    # grid_search_params = {'estimator': model,             # 目标分类器
    #                       'param_grid': parameters,  # 前面定义的我们想要优化的参数
    #                       'cv': ps,                     # 使用前面自定义的split验证策略
    #                       'n_jobs': -1,                 # 并行运行的任务数，-1表示使用所有CPU
    #                       }
    #
    # grsearch = GridSearchCV(**grid_search_params)
    # grsearch.fit(train_val_features, train_val_labels)
    #
    # print("best:",grsearch.best_params_)
    # print("socre:",grsearch.best_score_)