"""CIFAR-10/100 loading (no torchvision dependency) and IID / non-IID partitioning.

Non-IID schemes follow Section 5.3 of the paper:
  * CIFAR10  : p% of a worker's samples belong to one (dominant) class, the rest
               to the other classes. p = 10 is IID.
  * CIFAR100 : every worker lacks p classes. p = 0 is IID.
"""
import os
import pickle
import tarfile
import urllib.request

import numpy as np
import torch

URLS = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
}
FOLDERS = {"cifar10": "cifar-10-batches-py", "cifar100": "cifar-100-python"}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
MEAN = {"cifar10": (0.4914, 0.4822, 0.4465), "cifar100": (0.5071, 0.4865, 0.4409)}
STD = {"cifar10": (0.2470, 0.2435, 0.2616), "cifar100": (0.2673, 0.2564, 0.2762)}


def _find_or_download(name, data_dir):
    folder = FOLDERS[name]
    for cand in (os.path.join(data_dir, folder), os.path.join(data_dir, name, folder)):
        if os.path.isdir(cand):
            return cand
    os.makedirs(data_dir, exist_ok=True)
    archive = os.path.join(data_dir, os.path.basename(URLS[name]))
    if not os.path.exists(archive):
        print(f"Downloading {URLS[name]} -> {archive}")
        urllib.request.urlretrieve(URLS[name], archive + ".part")
        os.replace(archive + ".part", archive)
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir)
    return os.path.join(data_dir, folder)


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_cifar(name, data_dir):
    """Returns (x_train uint8 [N,3,32,32], y_train int64, x_test, y_test) as numpy arrays."""
    root = _find_or_download(name, data_dir)
    if name == "cifar10":
        batches = [_unpickle(os.path.join(root, f"data_batch_{i}")) for i in range(1, 6)]
        x_tr = np.concatenate([b["data"] for b in batches])
        y_tr = np.concatenate([b["labels"] for b in batches])
        test = _unpickle(os.path.join(root, "test_batch"))
        x_te, y_te = test["data"], np.array(test["labels"])
    else:
        train, test = _unpickle(os.path.join(root, "train")), _unpickle(os.path.join(root, "test"))
        x_tr, y_tr = train["data"], np.array(train["fine_labels"])
        x_te, y_te = test["data"], np.array(test["fine_labels"])
    x_tr = x_tr.reshape(-1, 3, 32, 32)
    x_te = x_te.reshape(-1, 3, 32, 32)
    return x_tr, y_tr.astype(np.int64), x_te, y_te.astype(np.int64)


def partition(labels, num_workers, num_classes, scheme, level, rng):
    """Returns a list of index arrays, one per worker (equal sizes)."""
    n = len(labels)
    per_worker = n // num_workers
    iid = scheme == "iid" or (scheme == "dominant" and level <= 100 / num_classes) or \
        (scheme == "missing" and level == 0)
    if iid:
        perm = rng.permutation(n)
        return [perm[i * per_worker:(i + 1) * per_worker] for i in range(num_workers)]

    pools = [rng.permutation(np.where(labels == c)[0]) for c in range(num_classes)]
    ptr = np.zeros(num_classes, dtype=int)
    remaining = lambda: np.array([len(p) for p in pools]) - ptr

    def take(c, k):
        k = min(k, remaining()[c])
        out = pools[c][ptr[c]:ptr[c] + k]
        ptr[c] += k
        return out

    def fill(allowed, need):
        # draw `need` samples from the allowed classes, proportionally to what is left
        out = []
        while need > 0:
            rem = remaining() * allowed
            if rem.sum() == 0:  # allowed classes exhausted: fall back to any class
                rem = remaining()
            counts = rng.multinomial(need, rem / rem.sum())
            for c in np.nonzero(counts)[0]:
                got = take(c, counts[c])
                out.append(got)
                need -= len(got)
        return out

    parts = []
    for w in range(num_workers):
        allowed = np.ones(num_classes)
        idx = []
        if scheme == "dominant":
            # p% of the samples from one dominant class, the rest from the other classes
            dom = w % num_classes
            idx.append(take(dom, int(per_worker * level / 100)))
            allowed[dom] = 0
        elif scheme == "missing":
            # the worker lacks `level` classes
            allowed[rng.choice(num_classes, size=int(level), replace=False)] = 0
        else:
            raise ValueError(f"unknown partition scheme {scheme}")
        idx += fill(allowed, per_worker - sum(len(i) for i in idx))
        parts.append(np.concatenate(idx))
    return parts


class GPUData:
    """Whole dataset kept on the device; batches sampled + augmented on the GPU."""

    def __init__(self, name, data_dir, num_workers, non_iid, seed, device):
        x_tr, y_tr, x_te, y_te = load_cifar(name, data_dir)
        self.num_classes = NUM_CLASSES[name]
        rng = np.random.default_rng(seed)
        scheme = "dominant" if name == "cifar10" else "missing"
        self.parts = partition(y_tr, num_workers, self.num_classes, scheme, non_iid, rng)
        mean = torch.tensor(MEAN[name], device=device).view(1, 3, 1, 1)
        std = torch.tensor(STD[name], device=device).view(1, 3, 1, 1)
        norm = lambda x: (torch.from_numpy(x).to(device).float() / 255.0 - mean) / std
        self.x_train, self.y_train = norm(x_tr), torch.from_numpy(y_tr).to(device)
        self.x_test, self.y_test = norm(x_te), torch.from_numpy(y_te).to(device)
        self.parts = [torch.from_numpy(p).to(device) for p in self.parts]
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    @property
    def samples_per_worker(self):
        return len(self.parts[0])

    def batch(self, worker, batch_size, augment=True):
        part = self.parts[worker]
        sel = part[torch.randint(len(part), (batch_size,), device=self.device, generator=self.gen)]
        x, y = self.x_train[sel], self.y_train[sel]
        if augment:  # random crop (pad 4) + horizontal flip
            x = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
            i, j = torch.randint(0, 9, (2,), generator=self.gen, device=self.device).tolist()
            x = x[:, :, i:i + 32, j:j + 32]
            flip = torch.rand(batch_size, device=self.device, generator=self.gen) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        return x, y
