import logging
import math
import time
from types import SimpleNamespace
from copy import deepcopy
from collections import defaultdict
from sparse import SparseLinear
import numpy as np
import torch
import wandb
from torchmetrics.classification import MulticlassCalibrationError
logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

class DL:
    def __init__(self, xs, ys, batch_size, device):
        self._xs = xs.to(device)
        self._ys = ys.to(device)
        self._bs = batch_size
        self._counter = 0

    def __iter__(self):
        return self

    def __next__(self):

        if self._counter * self._bs >= len(self._xs):
            perm = torch.randperm(self._xs.size(0))
            self._xs = self._xs[perm]
            self._ys = self._ys[perm]
            self._counter = 0
            raise StopIteration
        x_, y_ = self._xs[self._counter * self._bs: (self._counter + 1) * self._bs], \
            self._ys[self._counter * self._bs: (self._counter + 1) * self._bs]
        self._counter += 1
        return x_, y_

    def __len__(self):
        return self._xs.shape[0] // self._bs

    @property
    def dataset(self):
        return self._xs
def shrink(mul):
    def shrink_mask_constant(obj):
        if hasattr(obj, 'mask_constant'):
            obj.mask_constant = obj.mask_constant * mul
    return shrink_mask_constant
@torch.no_grad()
def get_pred(model,
             dl,
             device,
             test_batch_size=250,
             ):
    logits, labels = [], []

    for xb, yb in dl:

        xb, yb = xb.to(device), yb.to(device)
        bs = xb.shape[0]
        ybs, outs = [], []

        if not isinstance(model, list):
            test_batch_size = min(test_batch_size, model.num_test_samples)

            for _ in range(model.num_test_samples // test_batch_size):
                xb_new = xb.repeat(test_batch_size, *[1] * (xb.ndim - 1))
                out = model(xb_new).cpu().view(test_batch_size, bs, -1)
                outs.append(out)
            outs = torch.cat(outs, 0)

        else:
            outs = torch.stack([s(xb) for s in model]).cpu()

        out = torch.logsumexp(outs, 0).sub_(math.log(outs.size(0)))

        logits.append(out)
        labels.append(yb.cpu())

    return torch.cat(logits, 0), torch.cat(labels, 0)


@torch.no_grad()
def get_loss(model,
             loss_func,
             dl,
             device):
    logits, labels = get_pred(model, dl, device)
    metric = MulticlassCalibrationError(num_classes=10, n_bins=15, norm='l1')
    ece = metric(logits, labels)
    loss = loss_func(logits, labels, reduction="none")
    hits = (logits.argmax(-1) == labels).float()
    acc = hits.mean().item()*100
    er = 100 - acc
    loss = loss.mean().item()
    
    return {'loss': loss,
            'er': er, 'ece': ece}


@torch.no_grad()
def fit(model,
        data,
        loss_func,
        num_updates,
        keep_curve,
        device,
        log_steps=200,
        eval_log_steps=1000,
        num_burnin_steps=10000,
        mul = 1):

    models = []
    ts = []
    train_outputs = []
    valid_outputs = []

    data_iter = iter(data.train_dl)

    t0 = start_time = time.time()

    for i in range(1, num_updates + 1):

        if keep_curve and (i % eval_log_steps == 0 or i==1):
            model.eval()
            t0 = time.time()

            valid_output = get_loss(
                models if len(models) > 0 else model,
                loss_func,
                data.valid_dl,
                device,
            )

            valid_outputs.append(valid_output)

            t1 = time.time()

            print(
                f"EVALUATE2 VAL NLL {valid_output['loss']:.3f} "
                f"VAL ERR {valid_output['er']:.2f} "
                f"EVAL TIME {t1 - t0:.2f} sec")
        model.train()
        model.apply(shrink(mul))

        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(data.train_dl)
        xb, yb = map(lambda x: x.to(device, non_blocking=True),
                     (xb, yb))
        output = model.step(xb, yb, loss_func)
        output['t'] = time.time()-start_time
        train_outputs.append(output)
        if i % log_steps == 0:
            passed_time = time.time() - t0
            ts.append(passed_time / log_steps)

            rolling = defaultdict(list)
            for item in train_outputs[-log_steps:]:
                for k, v in item.items():
                    rolling[k].append(v)
            summary = ' '.join([f"{k}: {np.mean(v):.3f}"
                                for k, v in rolling.items()])

            logging.info(f"STEP {i} {summary} "
                         f"TIME/STEP {np.mean(ts[-log_steps:]):.3f} sec")
            wandb.log({ k:np.mean(v)
                                for k, v in rolling.items()})
            t0 = time.time()
            
    model.eval()
    t0 = time.time()

    test_results = get_loss(
        models if len(models)>0 else model,
        loss_func,
        data.valid_dl,
        device,
    )

    t1 = time.time()
    logging.info(f"Evaluation took {t1 - t0:.2f} sec.")
    eval_time =  time.time() - t1
    
    return SimpleNamespace(**test_results,
                           **{k:np.mean(v) for k, v in rolling.items()},
                           eval_time=eval_time,
                           train_outputs=train_outputs,
                           valid_outputs=valid_outputs)
