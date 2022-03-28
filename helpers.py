from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import pywt
from scipy import signal

def plot_confusion_matrix(true, preds, labels):
    confusion_FC = confusion_matrix(y_true=true, y_pred=preds, normalize='pred')

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_FC,
                                display_labels=labels)

    disp.plot()
    disp.ax_.set_xticklabels(labels, rotation=12)
    plt.title('Confusion Matrix of Shallow CNN by task')
    plt.show()

def conv2DPrepreprocessing(X, window):
    n, h, w, ch = X.shape

    if h % window != 0:
        raise Exception("Improper window size")

    out = []
    indices = np.arange(0, h + window, window)
    for sample in X:
        temp = None
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]
            sub_part = sample[start : end, :, :]
            if i == 0:
                temp = sub_part
            else:
                temp = np.hstack((temp, sub_part))
        out.append(temp)
    
    return np.array(out)

def sampleWindows(X, y, windowSize, nSamples):
    n, h, w, ch = X.shape
    if windowSize > h:
        raise Exception("Improper window size")

    totalPossible = h - windowSize + 1
    if totalPossible < nSamples:
        raise Exception("Too large nSamples asked")
    
    out = []
    labels = []
    end = h - windowSize
    findx = []
    for i, sample in enumerate(X):
        np.random.seed(42)
        indices = np.random.choice(end + 1, nSamples, replace=False)
    
        for id in indices:
            out.append(sample[id : (id + windowSize), :, :])
            labels.append(y[i])
            findx.append(i)
    
    out = np.array(out)
    labels = np.array(labels)
    findx = np.array(findx)

    out, labels, findx = shuffle(out, labels, findx)
    return out, labels, findx

def maxVoting(test, truelabels, indices, original_X, network):
  preds = np.argmax(network.predict(test), axis = 1)
  actual = []
  true = []
  for i in range(len(original_X)):
    ind = np.where(indices == i)

    temp_preds = preds[ind]

    vals, counts = np.unique(temp_preds, return_counts=True)
    mode_value = np.argwhere(counts == np.max(counts))[0]
    mode_value = int(np.squeeze(mode_value))

    r = vals[mode_value]
    actual.append(r)
    temp = np.argmax(truelabels[ind[0]], axis = 1)
    true.append(temp[0])

  return accuracy_score(true, actual)

def CWT(X, scales, waveletname='morl'):
    out = []
    k = 0
    n, h, w, ch = X.shape
    for sample in X:
        images = None
        for i in range(ch):
            coeff, freq = pywt.cwt(sample[:, :, i], scales, waveletname, 1)
            coeff_ = coeff[:,:(h - 1)]
            if i == 0:
                images = coeff_
            else:
                images = np.dstack((images, coeff_))

        out.append(images)
        k = k + 1
        print("Completed signals: ", k)
    
    return np.array(out)

def CWTScipy(X, fs = 250, w = 6):
    out = []
    n, h, w, ch = X.shape

    k = 0
    freq = np.linspace(1, fs/2, h)
    widths = w * fs / (2 * freq * np.pi)

    for sample in X:
        images = None
        for i in range(ch):
            sig = sample[:, :, i]
            coeff = np.abs(signal.cwt(sig.reshape(-1), signal.morlet2, widths))
            if i == 0:
                images = coeff
            else:
                images = np.dstack((images, coeff))

        out.append(images)
        k+= 1
        print("Completed signal: ", k)

    return np.array(out)