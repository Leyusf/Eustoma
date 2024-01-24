import eustoma


def accuracy(prediction, labels):
    labels, prediction = eustoma.as_variable(labels), eustoma.as_variable(prediction)

    pred = prediction.data.argmax(axis=1).reshape(labels.shape)
    result = (pred == labels.data)
    acc = result.mean()
    return eustoma.Variable(eustoma.as_array(acc))
