import numpy as np
import tensorflow as tf
from tensorflow_probability import edward2 as ed
import math
import wandb


def main(
    project_name=None,
    seed=None,
    dataset="fmnist",
    NLabels=10,
    algorithm=0,
    PARTICLES=20,
    batch_size=100,
    num_epochs=50,
    num_hidden_units=20,
    lamb=1.0,
    prior_scale=1.0,
):
    """Run experiments for MAP, Variational, PAC^2-Variational and PAC^2_T-Variational algorithms for the supervised classification task.
    Args:
        dataSource: The data set used in the evaluation.
        NLabels: The number of labels to predict.
        NPixels: The size of the images: NPixels\times NPixels.
        algorithm: Integer indicating the algorithm to be run.
            0- MAP Learning
            1- Variational Learning
            2- PAC^2-Variational Learning
            3- PAC^2_T-Variational Learning
        PARTICLES: Number of Monte-Carlo samples used to compute the posterior prediction distribution.
        batch_size: Size of the batch.
        num_epochs: Number of epochs.
        num_hidden_units: Number of hidden units in the MLP.
    Returns:
        NLL: The negative log-likelihood over the test data set.
    """
    run_name = f"{dataset}_{algorithm}_{lamb}_{prior_scale}"

    wandb.init(
        project=project_name,
        name=f"{run_name}",
        mode="online",
        config={
            "algorithm": algorithm,
            "seed": seed,
            "dataset": dataset,
            "batch_size": batch_size,
            "n_hidden": num_hidden_units,
            "n_epochs": num_epochs,
            "prior_scale": prior_scale,
            "lamb": lamb,
            "n_post_sampels": PARTICLES,
        },
    )

    np.random.seed(seed)
    tf.set_random_seed(seed)

    sess = tf.Session()

    if dataset == "fmnist":
        (x_train, y_train), (
            x_test,
            y_test,
        ) = tf.keras.datasets.fashion_mnist.load_data()
        NPixels = 28
    if dataset == "c10":
        (x_train, y_train), (
            x_test,
            y_test,
        ) = tf.keras.datasets.cifar10.load_data()
        x_train = sess.run(
            tf.cast(tf.squeeze(tf.image.rgb_to_grayscale(x_train)), dtype=tf.float32)
        )
        x_test = sess.run(
            tf.cast(tf.squeeze(tf.image.rgb_to_grayscale(x_test)), dtype=tf.float32)
        )
        NPixels = 32

    x_train, x_test = x_train / 255.0, x_test / 255.0

    N = x_train.shape[0]
    M = batch_size

    x_batch = tf.placeholder(
        dtype=tf.float32, name="x_batch", shape=[None, NPixels * NPixels]
    )
    y_batch = tf.placeholder(
        dtype=tf.float32,
        name="y_batch",
        shape=[
            None,
        ],
    )

    def model(NHIDDEN, x):
        W = ed.Normal(
            loc=tf.zeros([NPixels * NPixels, NHIDDEN]), scale=prior_scale, name="W"
        )
        b = ed.Normal(loc=tf.zeros([1, NHIDDEN]), scale=prior_scale, name="b")

        W_out = ed.Normal(
            loc=tf.zeros([NHIDDEN, NLabels]), scale=prior_scale, name="W_out"
        )
        b_out = ed.Normal(loc=tf.zeros([1, NLabels]), scale=prior_scale, name="b_out")

        hidden_layer = tf.nn.relu(tf.matmul(x, W) + b)
        out = tf.matmul(hidden_layer, W_out) + b_out
        y = ed.Categorical(logits=out, name="y")

        return W, b, W_out, b_out, x, y

    def qmodel(NHIDDEN):
        W_loc = tf.Variable(
            tf.random_normal([NPixels * NPixels, NHIDDEN], 0.0, 0.1, dtype=tf.float32)
        )
        b_loc = tf.Variable(tf.random_normal([1, NHIDDEN], 0.0, 0.1, dtype=tf.float32))

        if algorithm == 0:
            W_scale = 0.000001
            b_scale = 0.000001
        else:
            W_scale = tf.nn.softplus(
                tf.Variable(
                    tf.random_normal(
                        [NPixels * NPixels, NHIDDEN], -3.0, stddev=0.1, dtype=tf.float32
                    )
                )
            )
            b_scale = tf.nn.softplus(
                tf.Variable(
                    tf.random_normal([1, NHIDDEN], -3.0, stddev=0.1, dtype=tf.float32)
                )
            )

        qW = ed.Normal(W_loc, scale=W_scale, name="W")
        qW_ = ed.Normal(W_loc, scale=W_scale, name="W")

        qb = ed.Normal(b_loc, scale=b_scale, name="b")
        qb_ = ed.Normal(b_loc, scale=b_scale, name="b")

        W_out_loc = tf.Variable(
            tf.random_normal([NHIDDEN, NLabels], 0.0, 0.1, dtype=tf.float32)
        )
        b_out_loc = tf.Variable(
            tf.random_normal([1, NLabels], 0.0, 0.1, dtype=tf.float32)
        )
        if algorithm == 0:
            W_out_scale = 0.000001
            b_out_scale = 0.000001
        else:
            W_out_scale = tf.nn.softplus(
                tf.Variable(
                    tf.random_normal(
                        [NHIDDEN, NLabels], -3.0, stddev=0.1, dtype=tf.float32
                    )
                )
            )
            b_out_scale = tf.nn.softplus(
                tf.Variable(
                    tf.random_normal([1, NLabels], -3.0, stddev=0.1, dtype=tf.float32)
                )
            )

        qW_out = ed.Normal(W_out_loc, scale=W_out_scale, name="W_out")
        qb_out = ed.Normal(b_out_loc, scale=b_out_scale, name="b_out")

        qW_out_ = ed.Normal(W_out_loc, scale=W_out_scale, name="W_out")
        qb_out_ = ed.Normal(b_out_loc, scale=b_out_scale, name="b_out")

        return qW, qW_, qb, qb_, qW_out, qW_out_, qb_out, qb_out_

    W, b, W_out, b_out, x, y = model(num_hidden_units, x_batch)

    qW, qW_, qb, qb_, qW_out, qW_out_, qb_out, qb_out_ = qmodel(num_hidden_units)

    with ed.interception(ed.make_value_setter(W=qW, b=qb, W_out=qW_out, b_out=qb_out)):
        pW, pb, pW_out, pb_out, px, py = model(num_hidden_units, x)

    with ed.interception(
        ed.make_value_setter(W=qW_, b=qb_, W_out=qW_out_, b_out=qb_out_)
    ):
        pW_, pb_, pW_out_, pb_out_, px_, py_ = model(num_hidden_units, x)

    pylogprob = tf.expand_dims(py.distribution.log_prob(y_batch), 1)
    py_logprob = tf.expand_dims(py_.distribution.log_prob(y_batch), 1)

    logmax = tf.stop_gradient(tf.math.maximum(pylogprob, py_logprob) + 0.000001)
    logmax = tf.constant(-math.log(0.9999))
    logmean_logmax = tf.math.reduce_logsumexp(
        tf.concat([pylogprob - logmax, py_logprob - logmax], 1), axis=1
    ) - tf.log(2.0)
    alpha = tf.expand_dims(logmean_logmax, 1)

    if algorithm == 3:
        hmax = 2 * tf.stop_gradient(
            alpha / tf.math.pow(1 - tf.math.exp(alpha), 2)
            + tf.math.pow(tf.math.exp(alpha) * (1 - tf.math.exp(alpha)), -1)
        )
    else:
        hmax = 1.0

    var = 0.5 * (
        tf.reduce_mean(tf.exp(2 * pylogprob - 2 * logmax) * hmax)
        - tf.reduce_mean(tf.exp(pylogprob + py_logprob - 2 * logmax) * hmax)
    )

    datalikelihood = tf.reduce_mean(py.distribution.log_prob(y_batch))

    logprior = (
        tf.reduce_sum(pW.distribution.log_prob(pW.value))
        + tf.reduce_sum(pb.distribution.log_prob(pb.value))
        + tf.reduce_sum(pW_out.distribution.log_prob(pW_out.value))
        + tf.reduce_sum(pb_out.distribution.log_prob(pb_out.value))
    )

    entropy = (
        tf.reduce_sum(qW.distribution.log_prob(qW.value))
        + tf.reduce_sum(qb.distribution.log_prob(qb.value))
        + tf.reduce_sum(qW_out.distribution.log_prob(qW_out.value))
        + tf.reduce_sum(qb_out.distribution.log_prob(qb_out.value))
    )

    entropy = -entropy

    KL = (-entropy - logprior) / (N * lamb)

    if algorithm == 2 or algorithm == 3:
        elbo = datalikelihood + var - KL
    elif algorithm == 1:
        elbo = datalikelihood - KL
    elif algorithm == 0:
        elbo = datalikelihood + logprior / N

    optimizer = tf.train.AdamOptimizer(0.001)
    t = []
    train = optimizer.minimize(-elbo)
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(num_epochs + 1):
        perm = np.random.permutation(N)
        x_train = np.take(x_train, perm, axis=0)
        y_train = np.take(y_train, perm, axis=0)

        x_batches = np.array_split(x_train, N / M)
        y_batches = np.array_split(y_train, N / M)

        for j in range(N // M):
            batch_x = np.reshape(x_batches[j], [x_batches[j].shape[0], -1]).astype(
                np.float32
            )
            batch_y = np.reshape(
                y_batches[j],
                [
                    y_batches[j].shape[0],
                ],
            ).astype(np.float32)

            value, _ = sess.run(
                [elbo, train], feed_dict={x_batch: batch_x, y_batch: batch_y}
            )
            t.append(-value)

        if i % 5 == 0:
            wandb.log({f"mfvi/train/neg_elbo": t[-1]}, step=i)
            wandb.log(
                {
                    f"mfvi/train/data": sess.run(
                        datalikelihood, feed_dict={x_batch: batch_x, y_batch: batch_y}
                    )
                },
                step=i,
            )
            wandb.log(
                {
                    f"mfvi/train/var": sess.run(
                        var, feed_dict={x_batch: batch_x, y_batch: batch_y}
                    )
                },
                step=i,
            )
            wandb.log(
                {
                    f"mfvi/train/KL": sess.run(
                        KL, feed_dict={x_batch: batch_x, y_batch: batch_y}
                    )
                },
                step=i,
            )
            wandb.log(
                {
                    f"mfvi/train/entropy": sess.run(
                        entropy, feed_dict={x_batch: batch_x, y_batch: batch_y}
                    )
                },
                step=i,
            )

            M_test = 1000

            N_TEST = x_test.shape[0]
            x_batches = np.array_split(x_test, N_TEST / M_test)
            y_batches = np.array_split(y_test, N_TEST / M_test)

            NLL = 0

            for k in range(N_TEST // M_test):
                batch_x = np.reshape(x_batches[k], [x_batches[k].shape[0], -1]).astype(
                    np.float32
                )
                batch_y = np.reshape(
                    y_batches[k],
                    [
                        y_batches[k].shape[0],
                    ],
                ).astype(np.float32)
                y_pred_list = []
                for _ in range(PARTICLES):
                    y_pred_list.append(
                        sess.run(
                            tf.expand_dims(py.distribution.log_prob(batch_y), axis=1),
                            feed_dict={x_batch: batch_x},
                        )
                    )
                y_preds = np.concatenate(y_pred_list, axis=1)
                score = -tf.reduce_sum(
                    tf.math.reduce_logsumexp(y_preds, axis=1)
                    - tf.log(np.float32(PARTICLES))
                )
                score = sess.run(score)
                NLL = NLL + score

            wandb.log({f"mfvi/test/bayes_nll": NLL / N_TEST}, step=i)


if __name__ == "__main__":
    import fire
    import os

    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", default="dryrun")
    fire.Fire(main)