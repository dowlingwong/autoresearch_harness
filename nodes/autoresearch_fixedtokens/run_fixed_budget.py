"""Frozen parent of the Track G node autoresearch_fixedtokens (R1-HARNESS-ISOLATION).

Started by the runner as ``python3 run_fixed_budget.py --tokens {budget}`` (no shell), with
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` from the spec's isolation env (the variable torch 2.6
reads; the fork's ``PYTORCH_ALLOC_CONF`` is read only from torch 2.9 on).

FROZEN.  This process never imports the editable train.py.  It checks the data (the files prepared by
prepare_data.py against data/data_manifest.json, no unlisted file allowed; prepare.py reads them from
``data/`` through AUTORESEARCH_CACHE_DIR), starts the frozen worker ``ar_worker.py``, which runs train.py with
the fixed budget of ``--tokens`` training tokens (ar_budget.py: exit 3 if the budget is not used, 7 for a
request beyond it, 8 for the validation split while train.py runs, 4 for any other exception out of train.py)
and seeds torch from AR_TRAIN_SEED before it (autoresearch itself seeds 42 in train.py; the port moved that
into the frozen code).  AR_SPLIT_SEED has no effect: the validation shard is fixed by autoresearch.

Metric (computed here): autoresearch's bits per byte on the pinned validation shard, as
``prepare.evaluate_bpb`` defines it (``EVAL_TOKENS`` tokens in rows of ``MAX_SEQ_LEN``, special tokens of byte
length 0 excluded).  This process builds every validation batch with prepare's loader and the frozen tokenizer
before it starts the worker, with a fixed evaluation batch size (``EVAL_BATCH_SIZE``, not train.py's), then
sends each batch's input tokens to the worker, which writes ``model(x)`` (float32 logits of shape (B, T,
vocab)) into a CPU buffer this process owns (a sealed ``memfd`` the worker maps, passed over the socket with
``SCM_RIGHTS``), and computes the per-token cross-entropy and the byte counts here, with the targets it holds.
Only this process computes the metric; the worker returns logits.  There is **no CUDA IPC** in the result path
(condition 7 of R1-HARNESS-ISOLATION: CUDA IPC cannot cross the worker's user namespace on the Ada host/driver,
and it would need a shared ``/dev/shm``): the parent allocates the CPU buffer, snapshots it after each answer and
moves the copy to its own device when it has one.  Before the
sum, a causality probe compares the logits of the first validation batch with the logits of the same rows whose
suffix after a cut position was replaced: the logits before the cut must not change (exit 9 otherwise;
tolerance ``CAUSAL_ATOL``; equal infinities count as equal); the largest change is printed as ``AR_INFO
causal_probe_max_change=`` for the census.  After the sum the first batch is evaluated again and must give the
same logits within ``CAUSAL_ATOL`` (exit 10), so a model that changes itself during the evaluation is refused
(the worker also compares its weights' fingerprint before and after, exit 10).  The result line ``AR_METRIC
val_bpb=`` goes to the runner's result channel.

Known limits (see the port report's threat model): the worker runs as the same user, so it can read the
parquet files (including the validation shard) and answer this process's requests with logits that are not a
fixed model's (for example behave causally only on the probe batches, which it could recognise); the probe and
the repeat check narrow that, they do not close it.  The budget accounting runs in the worker.
"""
import ar_isolated as iso
CHANNEL = iso.parent_start()  # first statement: non-dumpable, then the result channel (R1-HARNESS-ISOLATION)

import math  # noqa: E402
import os  # noqa: E402

import ar_data  # noqa: E402
import ar_parent as P  # noqa: E402

EXIT_NOT_CAUSAL = 9
EXIT_LOGITS_CHANGED = 10
CAUSAL_ATOL = 0.05  # logits are soft-capped at 15; bf16 noise between two forwards is far below this
# Kit review round 1, M4: the evaluation batch size is fixed here, not taken from train.py, so the edited code
# cannot choose which validation rows are scored (or how the loader packs them).  It is the baseline train.py's
# DEVICE_BATCH_SIZE, 16, which divides the 10,240 validation rows (EVAL_TOKENS / MAX_SEQ_LEN): every row is
# scored, the same rows evaluate_bpb scores for the baseline.  (Fewer rows, as in the port's CPU smoke, give one
# batch of all rows.)
EVAL_BATCH_SIZE = 16
PROBE_CUTS = (0.25, 0.5, 0.75)  # fractions of the row length
REQUIRED_FILES = ["tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt", "data/shard_06542.parquet",
                  *[f"data/shard_{i:05d}.parquet" for i in range(10)]]
NODE = "autoresearch_fixedtokens"


class LogitSource:
    """The worker's logits for an input batch, through a CPU buffer this process owns.

    R1-TRACKG-AR-COND7 option A: the parent allocates the ``memfd`` CPU buffer (``iso.SharedBuffer``) and passes
    its descriptor to the worker with ``bind_cpu``; the worker maps it as CPU memory and copies the
    (bf16-autocast) ``model(x)`` output into it as float32.  This is the path on a GPU host too: after each
    answer the parent takes a private snapshot and moves it to its own device.  No CUDA IPC is used (it cannot
    cross the worker's user namespace on the Ada host/driver and would need a shared ``/dev/shm``).
    """

    def __init__(self, worker, shape, torch, device):
        self.worker, self.torch, self.device = worker, torch, device
        self.shape = tuple(int(d) for d in shape)
        self.shared = iso.SharedBuffer(self.shape, "float32")
        try:
            worker.conn.send("bind_cpu", {"buffer": self.shared.describe()}, {}, [self.shared.fd])
            op, meta, _arrays, fds = worker.conn.recv()
        except (iso.IsolationProtocolError, OSError) as error:
            P.fail(P.EXIT_WORKER, f"worker bind_cpu: {error}")
        for fd in fds:
            os.close(fd)
        if op != "ok":
            P.fail(P.EXIT_WORKER, f"worker bind_cpu: {meta.get('error', op)}")

    def __call__(self, x):
        """A private float32 copy of the worker's logits for x (B, T), on this process's device."""
        torch = self.torch
        tokens = x.detach().to("cpu", torch.int64).numpy()
        P.call(self.worker, "logits", {}, {"x": tokens})
        return torch.from_numpy(self.shared.snapshot()).to(self.device)


def causality_probe(logits_of, x, base, vocab: int, torch) -> float:
    """Logits before each cut must not depend on the tokens after it; returns the largest change seen."""
    T = x.shape[1]
    worst = 0.0
    if torch.isnan(base).any():
        return float("nan")  # a non-finite metric follows; nothing to compare
    for frac in PROBE_CUTS:
        cut = max(1, int(T * frac))
        altered = x.clone()
        altered[:, cut:] = torch.roll(x, shifts=1, dims=0)[:, cut:]
        if torch.equal(altered, x):
            altered[:, cut:] = (x[:, cut:] + 1) % vocab
        after = logits_of(altered)
        after_cut, before = after[:, :cut], base[:, :cut]
        change = torch.where(after_cut == before, torch.zeros_like(before), (after_cut - before).abs())  # inf == inf
        diff = change.nan_to_num(nan=float("inf")).max().item()
        worst = max(worst, diff)
        if not diff <= CAUSAL_ATOL:
            P.fail(EXIT_NOT_CAUSAL, f"logits before position {cut} changed by {diff:.4g} when later tokens "
                                    f"changed (tolerance {CAUSAL_ATOL}): the model is not causal")
    return worst


def main() -> None:
    budget = P.read_budget("tokens")
    raw_train = os.environ.get("AR_TRAIN_SEED")
    train_seed = int(raw_train) if raw_train is not None else 42
    print(f"AR_SEEDS train={train_seed}{'' if raw_train is not None else ' (default, autoresearch seed)'} "
          f"split={os.environ.get('AR_SPLIT_SEED')} (no effect: fixed validation shard)", flush=True)
    data_dir = ar_data.data_dir_of(__file__)
    ar_data.verify(data_dir, NODE, REQUIRED_FILES)
    os.environ["AUTORESEARCH_CACHE_DIR"] = str(data_dir.resolve())

    import torch
    import prepare  # frozen; this process never imports train.py
    from torch.nn.functional import cross_entropy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = prepare.Tokenizer.from_directory()
    vocab = int(tokenizer.get_vocab_size())
    max_seq_len, eval_tokens = prepare.MAX_SEQ_LEN, prepare.EVAL_TOKENS
    token_bytes = prepare.get_token_bytes(device=device)

    rows = eval_tokens // max_seq_len
    batch_size = min(EVAL_BATCH_SIZE, rows)
    if rows % batch_size:
        P.fail(P.EXIT_BAD_INVOCATION, f"EVAL_TOKENS / MAX_SEQ_LEN = {rows} rows is not a multiple of the evaluation batch "
                              f"size {batch_size}")
    steps = rows // batch_size
    # Kit review round 1, M3: every validation batch is read before the worker starts, so the editable code
    # cannot change the validation shard that is scored (the data are re-checked after the evaluation, too).
    val_loader = prepare.make_dataloader(tokenizer, batch_size, max_seq_len, "val")
    batches = []
    for _ in range(steps):
        x, y, _ = next(val_loader)
        batches.append((x.to("cpu").clone(), y.clone()))  # x and y are views of the loader's reused buffer
    del val_loader
    print(f"AR_INFO eval_rows={rows} eval_batch_size={batch_size}", flush=True)

    worker = P.start_worker()
    P.call(worker, "train", {"budget": budget, "train_seed": train_seed})
    logits_of = LogitSource(worker, (batch_size, max_seq_len, vocab), torch, device)

    total_nats, total_bytes = 0.0, 0
    first_x = first_logits = None
    with torch.no_grad():
        for index, (x, y) in enumerate(batches):
            logits = logits_of(x)
            if index == 0:
                first_x, first_logits = x.clone(), logits.clone()
                worst = causality_probe(logits_of, x, logits, vocab, torch)
                print(f"AR_INFO causal_probe_max_change={worst:.3g} (tolerance {CAUSAL_ATOL})", flush=True)
            nats = cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction="none")
            nbytes = token_bytes[y.reshape(-1)]
            total_nats += (nats * (nbytes > 0)).sum().item()
            total_bytes += nbytes.sum().item()
        again = logits_of(first_x)
        if not torch.isnan(first_logits).any():
            change = torch.where(again == first_logits, torch.zeros_like(again), (again - first_logits).abs())
            diff = change.nan_to_num(nan=float("inf")).max().item()
            if not diff <= CAUSAL_ATOL:
                P.fail(EXIT_LOGITS_CHANGED, f"the logits of the first validation batch changed by {diff:.4g} during "
                                            "the evaluation: the model changed itself")
    P.call(worker, "finish")
    P.stop_worker(worker)
    ar_data.verify(data_dir, NODE, REQUIRED_FILES, prefix="AR_ERROR train.py changed the data:")
    if torch.cuda.is_available():
        print(f"AR_INFO parent_peak_vram_mb={torch.cuda.max_memory_allocated() / 2**20:.1f}", flush=True)
    val_bpb = total_nats / (math.log(2) * total_bytes)
    CHANNEL.write(f"AR_METRIC val_bpb={val_bpb:.6f}")
    CHANNEL.close()


if __name__ == "__main__":
    main()
