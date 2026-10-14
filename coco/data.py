"""Dataset loading (CIFAR-10/100 without torchvision, UNSW-NB15) and IID / non-IID partitioning.

Partition schemes:
  * dominant  : p% of a worker's samples belong to one (dominant) class, the rest to the
                other classes (paper, CIFAR10; p = 100/num_classes is IID).
  * missing   : every worker lacks p classes (paper, CIFAR100; p = 0 is IID).
  * dirichlet : class proportions of each worker ~ Dir(alpha) (common for tabular /
                intrusion-detection FL; smaller alpha = more skewed, unequal sizes).
  * iid       : uniform random split.
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


def dirichlet_partition(labels, num_workers, num_classes, alpha, rng, min_size=50):
    """Splits every class among workers with proportions drawn from Dir(alpha)."""
    for _ in range(1000):
        parts = [[] for _ in range(num_workers)]
        for c in range(num_classes):
            idx = rng.permutation(np.where(labels == c)[0])
            props = rng.dirichlet(np.full(num_workers, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for w, chunk in enumerate(np.split(idx, cuts)):
                parts[w].append(chunk)
        parts = [rng.permutation(np.concatenate(p)) for p in parts]
        if min(len(p) for p in parts) >= min_size:
            return parts
    raise RuntimeError(f"could not give every worker >= {min_size} samples with alpha={alpha}")


def partition(labels, num_workers, num_classes, scheme, level, rng):
    """Returns a list of index arrays, one per worker (equal sizes except for dirichlet,
    where ``level`` is the concentration alpha)."""
    if scheme == "dirichlet":
        return dirichlet_partition(labels, num_workers, num_classes, level, rng)
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
    """Whole dataset kept on the device; batches sampled (and augmented for images) on the GPU."""

    def __init__(self, name, data_dir, num_workers, scheme, level, seed, device, unsw_task="multiclass"):
        if name == "unsw-nb15":
            from .unsw import load_unsw
            x_tr, y_tr, x_te, y_te, self.class_names = load_unsw(data_dir, unsw_task)
            self.num_classes = len(self.class_names)
            self.augment = False
            to_dev = lambda x: torch.from_numpy(x).to(device)
            self.x_train, self.x_test = to_dev(x_tr), to_dev(x_te)
        else:
            x_tr, y_tr, x_te, y_te = load_cifar(name, data_dir)
            self.num_classes = NUM_CLASSES[name]
            self.class_names = [str(c) for c in range(self.num_classes)]
            self.augment = True
            mean = torch.tensor(MEAN[name], device=device).view(1, 3, 1, 1)
            std = torch.tensor(STD[name], device=device).view(1, 3, 1, 1)
            norm = lambda x: (torch.from_numpy(x).to(device).float() / 255.0 - mean) / std
            self.x_train, self.x_test = norm(x_tr), norm(x_te)
        self.y_train, self.y_test = torch.from_numpy(y_tr).to(device), torch.from_numpy(y_te).to(device)
        self.input_dim = self.x_train.shape[1]

        rng = np.random.default_rng(seed)
        parts = partition(y_tr, num_workers, self.num_classes, scheme, level, rng)
        self.parts = [torch.from_numpy(np.asarray(p)).to(device) for p in parts]
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    @property
    def samples_per_worker(self):
        return float(np.mean([len(p) for p in self.parts]))

    def class_counts(self):
        """[num_workers, num_classes] label histogram of each worker's local data."""
        return np.stack([torch.bincount(self.y_train[p], minlength=self.num_classes).cpu().numpy()
                         for p in self.parts])

    def batch(self, worker, batch_size):
        part = self.parts[worker]
        sel = part[torch.randint(len(part), (batch_size,), device=self.device, generator=self.gen)]
        x, y = self.x_train[sel], self.y_train[sel]
        if self.augment:  # random crop (pad 4) + horizontal flip
            x = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
            i, j = torch.randint(0, 9, (2,), generator=self.gen, device=self.device).tolist()
            x = x[:, :, i:i + 32, j:j + 32]
            flip = torch.rand(batch_size, device=self.device, generator=self.gen) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
        return x, y
