import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import torch
from multiprocessing import Pool


def apply(func, args=None, kwds=None):
    """Launch a new process to call the function.

    This can be used to clear Tensorflow GPU memory after model execution:
    https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution
    """
    with Pool(1) as p:
        if args is None and kwds is None:
            r = p.apply(func)
        elif kwds is None:
            r = p.apply(func, args=args)
        elif args is None:
            r = p.apply(func, kwds=kwds)
        else:
            r = p.apply(func, args=args, kwds=kwds)
    return r


def gelu(x):
    return 0.5*x*(1+tf.math.tanh(tf.math.sqrt(2/np.pi)*(x+0.044715*x**3)))


def dirichlet(inputs, outputs):
    x_trunk = inputs[1]
    x, t = x_trunk[:, 0:1], x_trunk[:, 1:2]
    return 10 * x * (1 - x) * t * (outputs + 1)


def index_gen0():
    index_list = []
    for i in range(10):
        index_list.append(np.arange(i*10, i*10+5, 1))
    return np.concatenate(index_list, axis=0)


metrics_list = []
index = index_gen0()

train_data = np.load("dr_train_ls_0.1_101_101.npz")
test_data = np.load("dr_test_ls_0.1_101_101.npz")
X_test = (np.repeat(test_data["X_test0"], 10201, axis=0), np.tile(test_data["X_test1"], (100, 1)))
y_test = test_data["y_test"].reshape(-1, 1)

lr = 0.001


def train0(index):
    import deepxde as dde
    mean_l2_relative_error = lambda a, b: dde.metrics.mean_l2_relative_error(a.reshape(100, 10201),
                                                                             b.reshape(100, 10201))
    len_input = len(index)
    X_train = (np.repeat(train_data["X_train0"][index], 10201, axis=0), np.tile(train_data["X_train1"], (len_input, 1)))
    y_train = train_data["y_train"][index].reshape(-1, 1)
    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
    )

    net.apply_output_transform(dirichlet)

    data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=[mean_l2_relative_error])

    checker = dde.callbacks.ModelCheckpoint("model/model0.ckpt", save_better_only=False, period=1000)
    losshistory, train_state = model.train(epochs=20000, callbacks=[checker], batch_size=20000,)
    metrics = np.array(losshistory.metrics_test).reshape(-1, 1)
    np.savetxt('metrics0.txt', metrics)
    return metrics


metrics_test = apply(train0, (index,))
metrics_list.append(metrics_test)


def select(sensor_value, i, epochs):
    import deepxde as dde
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    X = geomtime.random_points(100000)

    def pde(x, y):
        dy = tf.gradients(y, x)[0]
        dy_x, dy_t = dy[:, 0:1], dy[:, 1:]
        dy_xx = tf.gradients(dy_x, x)[0][:, 0:1]
        z = tfp.math.batch_interp_regular_1d_grid(x[:, :1], 0, 1, np.float32(sensor_value))
        return dy_t - 0.01 * dy_xx - 0.01 * y ** 2 - z

    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
    )
    net.apply_output_transform(dirichlet)
    net.build()
    net.inputs = [sensor_value, None]
    data_pde = dde.data.TimePDE(geomtime, pde, [], num_domain=2500, num_test=10000)

    model = dde.Model(data_pde, net)
    model.compile("adam", lr=lr)
    model.restore(f"model/model{i-1}.ckpt-{epochs}.ckpt", verbose=1)
    residual = np.abs(model.predict(X, operator=pde)).mean()
    return residual


def train(index, i, epochs):
    import deepxde as dde
    mean_l2_relative_error = lambda a, b: dde.metrics.mean_l2_relative_error(a.reshape(100, 10201),
                                                                             b.reshape(100, 10201))
    len_input = len(index)
    X_train = (np.repeat(train_data["X_train0"][index], 10201, axis=0), np.tile(train_data["X_train1"], (len_input, 1)))
    y_train = train_data["y_train"][index].reshape(-1, 1)

    data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
    )
    net.apply_output_transform(dirichlet)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=[mean_l2_relative_error])
    checker = dde.callbacks.ModelCheckpoint(f"model/model{i}.ckpt", save_better_only=False, period=1000)

    if i == 19:
        losshistory, train_state = model.train(epochs=50000, callbacks=[checker], batch_size=20000, model_restore_path=f"model/model{i-1}.ckpt-{epochs}.ckpt")
    else:
        losshistory, train_state = model.train(epochs=epochs, callbacks=[checker], batch_size=20000, model_restore_path=f"model/model{i-1}.ckpt-{epochs}.ckpt")
    metrics = np.array(losshistory.metrics_test).reshape(-1, 1)
    np.savetxt(f'metrics{i}.txt', metrics)
    return metrics


epochs = 20000
for i in range(1, 20, 1):
    input_functions = train_data["X_train0"][i*50:(i+1)*50, :]
    residual_list = []
    for j in range(50):
        print(j)
        sensor_value = input_functions[j]
        residual = apply(select, (sensor_value, i, epochs))
        residual_list.append(residual)
    residual_list = np.array(residual_list)
    err_eq = torch.tensor(residual_list)
    x_ids = torch.topk(err_eq, 10, dim=0)[1].numpy() + 50*i
    index = np.concatenate([index, x_ids], 0)
    metrics_test = apply(train, (index, i, epochs))
    metrics_list.append(metrics_test)
print(index)
np.savetxt('index.txt', index)
metrics_list = np.concatenate(metrics_list, axis=0)
np.savetxt('metrics.txt', metrics_list)


