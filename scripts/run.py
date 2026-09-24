"""Run one class-incremental experiment and write its results as JSON.

    python scripts/run.py --dataset mnist --method ahr --seed 0
    python scripts/run.py --dataset cifar10 --method icarl --seed 0 --epochs 50

Methods
    ahr                AHR: latent exemplars, count = raw budget x input size / latent size
    ahr_lossless       AHR with perfect exemplars: same count as AHR, stored raw
    ahr_lossy_mini     AHR with as many (latent) exemplars as the raw-exemplar baselines
    ahr_lossless_mini  AHR with as many raw exemplars as the raw-exemplar baselines
    ft, ft_e, icarl    softmax baselines (see ahr/baselines.py)
    joint              upper bound
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ahr.baselines import JointLearner, SoftmaxLearner  # noqa: E402
from ahr.common import Timer, accuracy_report, set_seed  # noqa: E402
from ahr.data import CILBenchmark  # noqa: E402
from ahr.models import n_params  # noqa: E402
from ahr.strategy import AHR  # noqa: E402

# Appendix B / Table 4 of the paper.
PRESETS = {
    "mnist":    dict(epochs=40, batch_size=128, latent_dim=20, raw_exemplars=200, augment=False, bf16=0),
    "svhn":     dict(epochs=50, batch_size=128, latent_dim=307, raw_exemplars=200, augment=True, bf16=1),
    "cifar10":  dict(epochs=50, batch_size=128, latent_dim=307, raw_exemplars=200, augment=True, bf16=1),
    "cifar100": dict(epochs=50, batch_size=256, latent_dim=307, raw_exemplars=2000, augment=True, bf16=1),
}

# AHR hyper-parameters that the paper does not report; chosen on MNIST / a CIFAR-10
# validation run (see REPRODUCTION.md).
AHR_DEFAULTS = {
    "mnist":    dict(lam=0.3, alpha_z=0.01, alpha_x=1.0, rfa_zeta=1.0, rfa_steps=20000, rfa_target=5.0),
    "svhn":     dict(lam=1.0, alpha_z=0.01, alpha_x=1.0, rfa_zeta=1.0, rfa_steps=20000, rfa_target=20.0, memorize_steps=1500),
    "cifar10":  dict(lam=1.0, alpha_z=0.01, alpha_x=1.0, rfa_zeta=1.0, rfa_steps=20000, rfa_target=20.0, memorize_steps=1500),
    "cifar100": dict(lam=1.0, alpha_z=0.01, alpha_x=1.0, rfa_zeta=1.0, rfa_steps=20000, rfa_target=20.0, memorize_steps=1500),
}

METHODS = ["ahr", "ahr_lossless", "ahr_lossy_mini", "ahr_lossless_mini",
           "ft", "ft_e", "icarl", "joint"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(PRESETS))
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="", help="suffix for the result file name")
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--class-order", default="natural", choices=["natural", "icarl", "seed"])
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--stop-after", type=int, default=None, help="only learn the first N tasks")
    # optimisation (Table 4: Adam, lr 1e-3, momentum 0.9)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--augment", type=int, choices=[0, 1])
    ap.add_argument("--crop-pad", type=int, default=4)
    ap.add_argument("--bf16", type=int, choices=[0, 1], help="bf16 autocast (default: on for ResNet)")
    ap.add_argument("--eval-batch", type=int, default=1000)
    ap.add_argument("--flush-denormal", type=int, default=1, choices=[0, 1],
                    help="flush denormal floats to zero (avoids large CPU slowdowns)")
    ap.add_argument("--log-every", type=int, default=10)
    # memory
    ap.add_argument("--raw-exemplars", type=int, help="total raw exemplars of the baselines")
    ap.add_argument("--n-exemplars", type=int, help="override the number of stored exemplars")
    ap.add_argument("--decoder-in-budget", type=int, default=0, choices=[0, 1],
                    help="charge the decoder's parameters to AHR's memory budget (Table 3)")
    ap.add_argument("--latent-bits", type=int, default=32, choices=[8, 16, 32])
    # AHR
    ap.add_argument("--latent-dim", type=int)
    ap.add_argument("--enc-pool", type=int, default=4)
    ap.add_argument("--latent-kind", default="vector", choices=["vector", "spatial"],
                    help="vector: pooled features -> linear -> m; spatial: 1x1 conv of the 8x8 "
                    "feature map to round(m/64) channels (m ~ 64*round(m/64))")
    ap.add_argument("--decoder-width", type=float, default=1.0)
    ap.add_argument("--hf-transplant", type=float, default=0.0,
                    help="probability of adding a real new image's high-frequency residual to a "
                    "decoded exemplar")
    ap.add_argument("--hf-sigma", type=float, default=1.0)
    ap.add_argument("--input-blur", type=float, default=0.0,
                    help="sigma of a fixed Gaussian low-pass in front of the AHR encoder (0: off)")
    ap.add_argument("--lam", type=float)
    ap.add_argument("--alpha-z", type=float)
    ap.add_argument("--alpha-x", type=float)
    ap.add_argument("--distill-norm", default="sq", choices=["sq", "l2"])
    ap.add_argument("--alpha-kd", type=float, default=0.0,
                    help="weight of the KD loss on distances to the old CCEs (off by default)")
    ap.add_argument("--kd-scale", type=float, default=0.25,
                    help="KD logits are -||z-p||^2 / (kd_scale * rfa_target^2)")
    ap.add_argument("--memory-mode", default="frozen", choices=["reencode", "frozen"],
                    help="reencode: Alg. 4 re-encodes decoded old exemplars with the new "
                    "encoder every task; frozen: codes are kept as stored")
    ap.add_argument("--alpha-mem", type=float, default=1.0,
                    help="weight of ||psi(m) - psi_old(m)||^2 on stored codes m")
    ap.add_argument("--memorize-epochs", type=int, default=20,
                    help="decoder-only memorisation epochs over the stored exemplars (frozen codes)")
    ap.add_argument("--memorize-steps", type=int, default=None,
                    help="memorisation length in optimisation steps (overrides --memorize-epochs)")
    ap.add_argument("--memorize-lr", type=float, default=1e-3)
    ap.add_argument("--memorize-codes", type=int, default=0, choices=[0, 1],
                    help="also optimise the stored codes during memorisation")
    ap.add_argument("--lam-recon-new", type=float, default=1.0,
                    help="latent loss (x lambda) on reconstructions of the new samples")
    ap.add_argument("--latent-domain", default="input", choices=["input", "recon"],
                    help="input: L_z on phi(x) and test on phi(x) (paper); recon: L_z only on decoder "
                    "outputs (decoded exemplars, reconstructions of new samples) and test on "
                    "phi(psi(phi(x)))")
    ap.add_argument("--recon-new-source", default="current", choices=["current", "old"],
                    help="reconstructions from the HAE being trained (detached) or the previous one")
    ap.add_argument("--selection", default="herding", choices=["rank", "herding", "random"])
    ap.add_argument("--rfa-zeta", type=float)
    ap.add_argument("--rfa-mass", type=float, default=1.0)
    ap.add_argument("--rfa-dt", type=float, default=0.01)
    ap.add_argument("--rfa-steps", type=int)
    ap.add_argument("--rfa-damping", type=float, default=0.0)
    ap.add_argument("--rfa-softening", type=float, default=1e-3)
    ap.add_argument("--rfa-target", type=float, help="stop RFA once new CCEs are this far apart "
                    "from all other CCEs (<=0: run the full --rfa-steps)")
    ap.add_argument("--replay-sampling", default="union", choices=["union", "balanced"],
                    help="baselines: shuffle new data U exemplars (FACIL) or AHR-style balanced batches")
    # iCaRL
    ap.add_argument("--kd-lambda", type=float, default=1.0)
    ap.add_argument("--kd-temperature", type=float, default=2.0)
    args = ap.parse_args(argv)
    for k, v in {**PRESETS[args.dataset], **AHR_DEFAULTS[args.dataset]}.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    if args.rfa_target is not None and args.rfa_target <= 0:
        args.rfa_target = None
    args.augment = bool(args.augment)
    args.bf16 = bool(args.bf16)
    return args


def exemplar_count(args, bench, decoder_params):
    raw = args.raw_exemplars
    if args.n_exemplars is not None:
        return args.n_exemplars
    if args.method in ("ft_e", "icarl", "ahr_lossless_mini", "ahr_lossy_mini"):
        return raw
    if args.method in ("ahr", "ahr_lossless"):
        budget = raw * bench.input_size  # scalars available to raw-exemplar baselines
        if args.decoder_in_budget:
            budget -= decoder_params
        return budget // args.latent_dim
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_flush_denormal(bool(args.flush_denormal))
    set_seed(args.seed)
    bench = CILBenchmark(args.dataset, args.data_root, args.n_tasks, args.class_order,
                         args.seed, args.train_fraction)
    name = f"{args.method}_s{args.seed}{('_' + args.tag) if args.tag else ''}"
    out_dir = os.path.join(args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, name + ".log")
    log_f = open(log_path, "w")

    t_start = time.time()

    def log(msg):
        if msg.startswith("  task"):
            msg += f" [{time.time() - t_start:.0f}s]"
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    args.memory_kind = "raw" if args.method in ("ahr_lossless", "ahr_lossless_mini") else "latent"
    if args.method.startswith("ahr"):
        from ahr.models import make_hae
        if args.latent_kind == "spatial" and args.dataset != "mnist":
            args.latent_dim = 64 * max(1, round(args.latent_dim / 64))
        dec_params = n_params(make_hae(args.dataset, bench.input_shape, args.latent_dim,
                                       args.decoder_width, args.enc_pool,
                                       latent_kind=args.latent_kind).decoder)
    else:
        dec_params = 0
    args.n_exemplars = exemplar_count(args, bench, dec_params)
    log(f"benchmark {args.dataset}({bench.n_tasks}/{bench.classes_per_task}) "
        f"input={bench.input_shape} method={args.method} seed={args.seed}")
    log("args: " + json.dumps(vars(args), sort_keys=True))

    set_seed(args.seed)
    timer = Timer()
    acc_matrix, accs = [], []
    extra = {}
    if args.method == "joint":
        learner = JointLearner(bench, args, log)
        learner.learn_all()
        x, y = bench.test_upto(bench.n_tasks - 1)
        acc, per_task = accuracy_report(learner.predict(x), y, bench.classes_per_task, bench.n_tasks)
        accs, acc_matrix = [acc], [per_task]
        log(f"joint: acc={acc:.2f} per-task={[round(a, 1) for a in per_task]} [{timer():.0f}s]")
    else:
        if args.method.startswith("ahr"):
            learner = AHR(bench, args, log)
        else:
            learner = SoftmaxLearner(bench, args, args.method, log)
        psnr = []
        for t, task in enumerate(bench.tasks[:args.stop_after]):
            learner.learn_task(t, task)
            x, y = bench.test_upto(t)
            acc, per_task = accuracy_report(learner.predict(x), y, bench.classes_per_task, t + 1)
            accs.append(acc)
            acc_matrix.append(per_task)
            msg = f"after task {t + 1}: acc={acc:.2f} per-task={[round(a, 1) for a in per_task]}"
            if args.method in ("ahr", "ahr_lossy_mini"):
                p, orig, dec = learner.memory_fidelity(n=2000)
                psnr.append(p)
                acc_rec, acc_mem = learner.diagnostics(x, y)
                msg += f" memory-PSNR={p:.2f}dB acc(recon test)={acc_rec:.1f} acc(memory)={acc_mem:.1f}"
                torch.save({"orig": orig, "dec": dec}, os.path.join(out_dir, f"{name}_samples_t{t + 1}.pt"))
            log(msg + f" [{timer():.0f}s]")
        if psnr:
            extra["memory_psnr"] = psnr
        if args.method.startswith("ahr"):
            extra["cces"] = learner.cces.tolist() if args.latent_dim <= 32 else None
            extra["memory_scalars"] = learner.memory.scalars()
            extra["memory_bytes"] = learner.memory.nbytes()
            extra["decoder_params"] = dec_params
            extra["encoder_params"] = n_params(learner.model.encoder)
        elif learner.memory is not None:
            extra["memory_scalars"] = learner.memory.scalars()
            extra["memory_bytes"] = learner.memory.nbytes()
    res = dict(dataset=args.dataset, method=args.method, seed=args.seed, tag=args.tag,
               final_acc=accs[-1], avg_inc_acc=sum(accs) / len(accs), accs=accs,
               acc_matrix=acc_matrix, time_s=timer(), n_exemplars=args.n_exemplars,
               args=vars(args), **extra)
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"FINAL {args.dataset} {args.method} seed={args.seed}: final acc={accs[-1]:.2f} "
        f"avg inc acc={res['avg_inc_acc']:.2f} time={timer() / 60:.1f}min")
    return res


if __name__ == "__main__":
    main()
