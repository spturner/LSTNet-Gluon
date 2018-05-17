# As train.py but with adjustments to run via SageMaker
# Inspired by https://github.com/cyrusmvahid/sagemaker-demos/blob/master/ml-workshop-day2.md

from __future__ import print_function

import argparse
import boto3
import logging
import mxnet as mx
from mxnet import nd, gluon, autograd

from dataset import TimeSeriesData
from model import LSTNet

logging.basicConfig(level=logging.DEBUG)


def train(channel_input_dirs, hyperparameters, hosts, num_gpus, **kwargs):
    # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
    # the current container environment, but here we just use simple cpu context.

    # retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get('batch_size', 128)
    epochs = hyperparameters.get('epochs', 100)
    learning_rate = hyperparameters.get('learning_rate', 0.1)
    clip_gradient = hyperparameters.get('clip_gradient', 10.)
    conv_hid = hyperparameters.get('conv_hid', 100)
    gru_hid = hyperparameters.get('gru_hid', 100)
    skip_gru_hid = hyperparameters.get('skip_gru_hid', 5)
    skip = hyperparameters.get('skip', 24)
    ar_window = hyperparameters.get('ar_window', 24)
    log_interval = hyperparameters.get('log_interval', 100)
    horizon = hyperparameters.get('horizon', 24)
    window = hyperparameters.get('window', 24*7)
    file_path = channel_input_dirs['training']

    ts_data = TimeSeriesData(file_path, window=window, horizon=horizon)
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()

    net = LSTNet(
        num_series=ts_data.num_series,
        conv_hid=conv_hid,
        gru_hid=gru_hid,
        skip_gru_hid=skip_gru_hid,
        skip=skip,
        ar_window=ar_window)
    l1 = gluon.loss.L1Loss()

    net.initialize(init=mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=ctx)

    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'

    trainer = gluon.Trainer(net.collect_params(),
                            optimizer='adam',
                            optimizer_params={'learning_rate': learning_rate, 'clip_gradient': clip_gradient},
                            kvstore=kvstore)

    train_data_loader = gluon.data.DataLoader(
        ts_data.train, batch_size=batch_size, shuffle=True, num_workers=16, last_batch='discard')

    scale = nd.array(ts_data.scale, ctx=ctx)

    loss = None
    print("Training Start")
    for e in range(epochs):
        epoch_loss = mx.nd.zeros((1,), ctx=ctx)
        num_iter = 0
        for data, label in train_data_loader:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            if loss is not None:
                loss.wait_to_read()
            with autograd.record():
                y_hat = net(data)
                loss = l1(y_hat * scale, label * scale)
            loss.backward()
            trainer.step(batch_size)
            epoch_loss = epoch_loss + loss.mean()
            num_iter += 1
        print("Epoch {:3d}: loss {:.4}".format(e, epoch_loss.asscalar() / num_iter))

    #net.save_params(out_path)
    print("Training End")
    return 0


def save(net, model_dir):
    # save the model
    y = net(mx.sym.var('data'))
    y.save('%s/model.json' % model_dir)
    net.collect_params().save('%s/model.params' % model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
    parser.add_argument('--data', type=str, required=True,
                        help='path of the data file')
    parser.add_argument('--out', type=str, required=True,
                        help='path of the trained network output')
    args = parser.parse_args()

    exit(train(args.data, args.out))
