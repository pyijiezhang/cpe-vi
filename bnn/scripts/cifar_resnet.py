from collections import defaultdict
import json
import os
import pickle
import wandb
from tqdm import tqdm, trange

import torch
import torch.distributions as dist
import torch.utils.data as data

import torchvision
import torchvision.transforms as tf


import bnn
from bnn.calibration import calibration_curve, expected_calibration_error as ece


STATS = {
    "CIFAR10": {
        "mean": (0.49139968, 0.48215841, 0.44653091),
        "std": (0.24703223, 0.24348513, 0.26158784),
    },
    "CIFAR100": {
        "mean": (0.50707516, 0.48654887, 0.44091784),
        "std": (0.26733429, 0.25643846, 0.27615047),
    },
}
ROOT = os.environ.get("DATASETS_PATH", "./data")
NUM_BINS = 10


def reset_cache(module):
    if hasattr(module, "reset_cache"):
        module.reset_cache()


def main(
    project_name=None,
    output_dir=None,
    cifar=10,
    resnet=18,
    lamb=1.0,
    prior_scale=1.0,
    aug=False,
    inference_config="mfvi",
    obj_pacbayes="elbo",
    seed=42,
    num_epochs=100,
    annealing_epochs=0,
    ml_epochs=0,
    train_samples=1,
    test_samples=5,
    lr=1e-3,
    batch_size=128,
    optimizer="adam",
    momentum=0.9,
    milestones=None,
    gamma=0.1,
    verbose=False,
    progress_bar=False,
):
    run_name = (
        f"{obj_pacbayes}_{aug}_{lamb}_{prior_scale}_{num_epochs}_{annealing_epochs}"
    )

    wandb.init(
        project=project_name,
        name=f"{run_name}",
        mode="online",
        config={
            "seed": seed,
            "num_epochs": num_epochs,
            "inference_config": inference_config,
            "obj_pacbayes": obj_pacbayes,
            "ml_epochs": ml_epochs,
            "lamb": lamb,
            "prior_scale": prior_scale,
            "aug": aug,
            "annealing_epochs": annealing_epochs,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "lr": lr,
            "cifar": cifar,
            "resnet": resnet,
        },
    )

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up data loaders
    dataset_name = f"CIFAR{cifar}"
    dataset_cls = getattr(torchvision.datasets, dataset_name)
    root = f"{ROOT}/{dataset_name.lower()}"
    print(f"Loading dataset {dataset_cls} from {root}")
    aug_tf = [
        tf.RandomCrop(32, padding=4, padding_mode="reflect"),
        tf.RandomHorizontalFlip(),
    ]
    norm_tf = [tf.ToTensor(), tf.Normalize(**STATS[dataset_name])]

    if aug:
        train_data = dataset_cls(
            root, train=True, transform=tf.Compose(aug_tf + norm_tf), download=True
        )
    else:
        train_data = dataset_cls(
            root, train=True, transform=tf.Compose(norm_tf), download=True
        )
    test_data = dataset_cls(
        root, train=False, transform=tf.Compose(norm_tf), download=True
    )

    train_loader = data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = data.DataLoader(test_data, batch_size=10000)

    net = bnn.nn.nets.make_network(
        f"resnet{resnet}", kernel_size=3, remove_maxpool=True, out_features=cifar
    )
    if inference_config == "mfvi":
        # set up net and optimizer
        with open("configs/ffg_u_cifar10.json") as f:
            cfg = json.load(f)
        cfg["prior_sd"] = prior_scale
        bnn.bayesianize_(net, **cfg)

    if verbose:
        print(net)
    net.to(device)

    if optimizer == "adam":
        optim = torch.optim.Adam(net.parameters(), lr)
    elif optimizer == "sgd":
        optim = torch.optim.SGD(net.parameters(), lr, momentum=momentum)
    else:
        raise RuntimeError("Unknown optimizer:", optimizer)

    # set up dict for tracking losses and load state dicts if applicable
    metrics = defaultdict(list)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        snapshot_sd_path = os.path.join(output_dir, "snapshot_sd.pt")
        snapshot_optim_path = os.path.join(output_dir, "snapshot_optim.sd")
        metrics_path = os.path.join(output_dir, "metrics.pkl")
        if os.path.isfile(snapshot_sd_path):
            net.load_state_dict(torch.load(snapshot_sd_path, map_location=device))
            optim.load_state_dict(torch.load(snapshot_optim_path, map_location=device))
            with open(metrics_path, "rb") as f:
                metrics = pickle.load(f)
        else:
            torch.save(net.state_dict(), os.path.join(output_dir, "initial_sd.pt"))
    else:
        snapshot_sd_path = None
        snapshot_optim_path = None
        metrics_path = None

    last_epoch = len(metrics["acc"]) - 1

    if milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones, gamma=gamma, last_epoch=last_epoch
        )
    else:
        scheduler = None

    kl_factor = 0.0 if ml_epochs > 0 or annealing_epochs > 0 else 1 / lamb
    annealing_rate = (lamb * annealing_epochs) ** -1 if annealing_epochs > 0 else 1.0

    epoch_iter = (
        trange(last_epoch + 1, num_epochs + 1, desc="Epochs")
        if progress_bar
        else range(last_epoch + 1, num_epochs + 1)
    )
    for i in epoch_iter:
        net.train()
        net.apply(reset_cache)
        batch_iter = (
            tqdm(iter(train_loader), desc="Batches")
            if progress_bar
            else iter(train_loader)
        )
        avg_nll_epoch = 0
        avg_var_epoch = 0
        for _, (x, y) in enumerate(batch_iter):
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            avg_nll = 0.0
            avg_var = 0.0
            for k in range(train_samples):
                # variance term
                logp1 = (
                    torch.distributions.Categorical(logits=net(x))
                    .log_prob(y)
                    .unsqueeze(1)
                )
                logp2 = (
                    torch.distributions.Categorical(logits=net(x))
                    .log_prob(y)
                    .unsqueeze(1)
                )
                logmax = (torch.max(logp1, logp2) + 0.1).detach()
                logp3 = torch.stack([logp1, logp2], 1).squeeze(-1)
                logmean = torch.logsumexp(logp3, 1) - torch.log(
                    torch.tensor(2.0, dtype=float)
                )
                inc = torch.unsqueeze(logmean, 1) - logmax
                hmax = (
                    2
                    * (
                        inc / torch.pow(1 - torch.exp(inc), 2)
                        + torch.pow(torch.exp(inc) * (1 - torch.exp(inc)), -1)
                    ).detach()
                )
                var_term = 0.5 * (
                    torch.sum(torch.exp(2 * logp1 - 2 * logmax) * hmax)
                    - torch.sum(torch.exp(logp1 + logp2 - 2 * logmax) * hmax)
                )

                yhat = net(x)
                nll = -dist.Categorical(logits=yhat).log_prob(y).mean() / train_samples
                if k == 0:
                    kl = torch.tensor(0.0, device=device)
                    for module in net.modules():
                        if hasattr(module, "parameter_loss"):
                            kl = kl + module.parameter_loss().sum()
                    metrics["kl"].append(kl.item())

                    if obj_pacbayes == "elbo":
                        loss = nll + kl * kl_factor / len(train_data)
                    if obj_pacbayes == "pacelbo":
                        loss = nll + kl * kl_factor / len(train_data) - var_term
                else:
                    if obj_pacbayes == "elbo":
                        loss = nll
                    if obj_pacbayes == "pacelbo":
                        loss = nll - var_term

                avg_nll += nll.item()
                avg_var += var_term.item()
                loss.backward(retain_graph=train_samples > 1)

            optim.step()

            net.apply(reset_cache)

            metrics["nll"].append(avg_nll)

            avg_nll_epoch += avg_nll
            avg_var_epoch += avg_var

        kl_elbo = kl.item() / len(train_data)
        actual_kl = kl_elbo * kl_factor
        nll_elbo = avg_nll_epoch / (len(train_data) / batch_size)
        var_elbo = avg_var_epoch / (len(train_data) / batch_size)
        neg_elbo = nll_elbo + kl_elbo / lamb
        neg_pacelbo = neg_elbo - var_elbo

        wandb.log({f"mfvi/train/kl": kl_elbo}, step=i)
        wandb.log({f"mfvi/train/actual_kl": actual_kl}, step=i)
        wandb.log({f"mfvi/train/gibbs_nll": nll_elbo}, step=i)
        wandb.log({f"mfvi/train/var": var_elbo}, step=i)
        wandb.log({f"mfvi/train/neg_elbo": neg_elbo}, step=i)
        wandb.log({f"mfvi/train/neg_pacelbo": neg_pacelbo}, step=i)

        if scheduler is not None:
            scheduler.step()

        if i % 5 == 0:
            net.eval()
            with torch.no_grad():
                ens_logits = []
                for _ in range(test_samples):
                    all_logits = []
                    all_Y = []
                    for X, Y in tqdm(test_loader, leave=False):
                        X, Y = X.to(device), Y.to(device)
                        _logits = net(X)
                        all_logits.append(_logits)
                        all_Y.append(Y)
                    all_logits = torch.cat(all_logits)
                    all_Y = torch.cat(all_Y)
                    ens_logits.append(all_logits)
                ens_logits = torch.stack(ens_logits)
                probs = ens_logits.mean(0).softmax(-1).detach()
                log_p = torch.distributions.Categorical(logits=ens_logits).log_prob(
                    all_Y
                )
                gibbs_loss = -log_p.mean()
                bayes_loss = (
                    torch.log(torch.tensor(log_p.shape[0])) - torch.logsumexp(log_p, 0)
                ).mean()
                Y_pred = ens_logits.softmax(dim=-1).mean(dim=0).argmax(dim=-1)
                acc = (Y_pred == all_Y).sum().item() / Y_pred.size(0)
                p, f, w = calibration_curve(
                    probs.cpu().numpy(), all_Y.cpu().numpy(), NUM_BINS
                )
                metrics["ece"].append(ece(p, f, w).item())

                wandb.log({f"mfvi/test/acc": acc}, step=i)
                wandb.log({f"mfvi/test/ece": ece(p, f, w).item()}, step=i)
                wandb.log({f"mfvi/test/gibbs_nll": gibbs_loss}, step=i)
                wandb.log({f"mfvi/test/bayes_nll": bayes_loss}, step=i)

                # if verbose:
                #     print(
                #         f"Epoch {i} -- Accuracy: {100 * metrics['acc'][-1]:.2f}%; ECE={100 * metrics['ece'][-1]:.2f}"
                #     )

                # free up some variables for garbage collection to save memory for smaller GPUs
                del probs
                del all_Y
                del p
                del f
                del w
                torch.cuda.empty_cache()

        # if output_dir is not None:
        #     torch.save(net.state_dict(), snapshot_sd_path)
        #     torch.save(optim.state_dict(), snapshot_optim_path)
        #     with open(metrics_path, "wb") as fn:
        #         pickle.dump(metrics, fn)

        if i >= ml_epochs:
            kl_factor = min(1 / lamb, kl_factor + annealing_rate)
            wandb.log({f"mfvi/train/kl_factor": kl_factor}, step=i)
            wandb.log({f"mfvi/train/lamb": 1 / kl_factor}, step=i)

    # print(f"Final test accuracy: {100 * metrics['acc'][-1]:.2f}")

    # if output_dir is not None:
    #     bin_width = NUM_BINS**-1
    #     bin_centers = np.linspace(bin_width / 2, 1 - bin_width / 2, NUM_BINS)

    #     plt.figure(figsize=(5, 5))
    #     plt.plot([0, 100], [0, 100], color="black", linestyle="dashed", alpha=0.5)
    #     plt.plot(100 * p[w > 0], 100 * f[w > 0], marker="o", markersize=8)
    #     plt.bar(
    #         100 * bin_centers[w > 0], 100 * w[w > 0], width=100 * bin_width, alpha=0.5
    #     )
    #     plt.xlabel("Mean probability predicted")
    #     plt.ylabel("Empirical accuracy")
    #     plt.title(
    #         f"Calibration curve (Accuracy={100 * metrics['acc'][-1]:.2f}%; ECE={100 * metrics['ece'][-1]:.4f})"
    #     )
    #     plt.savefig(os.path.join(output_dir, "calibration.png"), bbox_inches="tight")

    wandb.alert(
        title=f"run_{run_name} finishes!",
        text=f"run_{run_name} finishes!",
        level=wandb.AlertLevel.WARN,
    )

    wandb.finish()


if __name__ == "__main__":
    import fire
    import os

    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", default="dryrun")
    fire.Fire(main)