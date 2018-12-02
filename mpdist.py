import pp, time
import numpy as np


def mpdist(X, Y, metric, dpdcs, verbose=False):
    m, n = len(X), len(Y)
    ppservers = ()
    job_server = pp.Server(ppservers=ppservers)
    if verbose:
        print "Starting pp with", job_server.get_ncpus(), "workers"

    start_time = time.time()

    res = np.empty((m, n), dtype=np.float32)
    for i in xrange(len(X)):
        jobs = [(j, job_server.submit(metric, (X[i], Y[j]), (), dpdcs)) for j in xrange(len(Y))]
        res[i, :] = [job() for j, job in jobs]

    if verbose:
        print "Time elapsed: ", time.time() - start_time, "s"
        job_server.print_stats()

    return res
