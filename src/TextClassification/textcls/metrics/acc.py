def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def mean_accuracy(preds, labels):
    dic = {}
    for pre, lb in zip(preds, labels):
        lb = int(lb)
        if lb not in dic:
            dic[lb] = [0, 0]
        dic[lb][1] += 1
        dic[lb][0] += pre==lb
    dic = {k: v[0]/v[1] for k, v in dic.items()}
    mean_acc = sum(dic.values())/len(dic)
    dic["mean_acc"] = mean_acc
    dic = {k: round(v, 5) for k, v in dic.items()}
    return mean_acc