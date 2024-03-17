import numpy as np

y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30,0.7,1,0,0.5])

def MAE(y_predicted, y_true):
    total_error = 0
    for yp, yt in zip(y_predicted, y_true):
        total_error += abs(yp - yt)
    print("Total Error is: ", total_error)
    mae = total_error/len(y_predicted)
    print("MAE: ", mae)

MAE(y_predicted, y_true)

np.abs(y_predicted - y_true)
np.mean(np.abs(y_predicted - y_true))

def MAE_np(y_predicted, y_true):
    return np.mean(np.abs(y_predicted - y_true))

MAE_np(y_predicted, y_true)

# Implement Log Loss or Binary Cross Entropy
np.log([0])
np.log([0.0000000000000000000000000000000001])

eplison = 1e - 15
np.log([1e-15])

y_predicted

y_predicted_new = [min(i, 1-(1e-15)) for i in y_predicted]
y_predicted_new

y_predicted_new = np.array(y_predicted_new)
y_predicted_new

-np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))

def log_loss(y_true, y_predicted):
    y_predicted_new = [max(i, (1e-15)) for i in y_predicted]
    y_predicted_new = [min(i, 1-(1e-15)) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))

log_loss(y_true, y_predicted)


y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30,0.7,1,0,0.5])

def mse(y_true, y_predicted):
    total_error = 0
    for yt, yp in zip(y_true, y_predicted):
        total_error += (yt-yp)**2
    print("Total Squared Error:",total_error)
    mse = total_error/len(y_true)
    print("Mean Squared Error:",mse)
    return mse
    
mse(y_true, y_predicted)
np.mean(np.square(y_true-y_predicted))

