import numpy as np

class BruteForceBLAS():
    """kNN search that uses a linear scan = brute force."""
    def __init__(self, metric = 'angular', precision=np.float32):
        if metric not in ('angular', 'euclidean'):
            raise NotImplementedError("BruteForceBLAS doesn't support metric %s" % metric)
        self._metric = metric
        self._precision = precision

    def fit(self, X):
        lens = (X ** 2).sum(-1)  # precompute (squared) length of each vector
        if self._metric == 'angular':
            X /= np.sqrt(lens)[..., np.newaxis]  # normalize index vectors to unit length
            self.index = np.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == 'euclidean':
            self.index = np.ascontiguousarray(X, dtype=self._precision)
            self.lengths = np.ascontiguousarray(lens, dtype=self._precision)

    def get_all_dists(self, query_id):
        v = self.index[query_id] # normalized
        if self._metric == 'angular':
            dists = -np.dot(self.index, v)
        elif self._metric == 'euclidean':
            dists = self.lengths - 2 * np.dot(self.index, v)
        return dists

    def check_rank(self, query_id, same_ids):
        # same_ids is the list of the ids in the same class
        v = self.index[query_id] # normalized
        if self._metric == 'angular':
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)
            dists = -np.dot(self.index, v)
        elif self._metric == 'euclidean':
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab
            dists = self.lengths - 2 * np.dot(self.index, v)
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!
        # this rank should start from 1, because of the self-retrieval 
        neighbor_dists = dists[same_ids]
        rank = np.count_nonzero( dists < neighbor_dists.min() ) 
        return rank

    def query(self, v, n, margin_ratio=10 ):
        n = min( n, self.index.shape[0] )
        knn_margin = n*margin_ratio
        knn = n + knn_margin
        knn = min( knn, self.index.shape[0] )

        """Find indices of `n` most similar vectors from the index to query vector `v`."""
        v = np.ascontiguousarray(v, dtype=self._precision)  # use same precision for query as for index
        # HACK we ignore query length as that's a constant not affecting the final ordering
        if self._metric == 'angular':
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)
            dists = -np.dot(self.index, v)
        elif self._metric == 'euclidean':
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab
            dists = self.lengths - 2 * np.dot(self.index, v)
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!

        # partition-sort by distance, get `n` closest
        indices = np.argpartition(dists, kth= knn-1 )[:knn] # kth starts from 0 index
        indices = sorted(indices, key=lambda index: dists[index])  # sort `n` closest into correct order

        cutoff = n
        last_dist = dists[ indices[n - 1] ] # distance of last item
        for i in range(n, knn):
            if last_dist == dists[ indices[i] ]:
                cutoff = cutoff + 1
            else:
                break
        retreived_dists = [ dists[i] for i in indices[:cutoff] ]
        amp = np.sqrt((v ** 2).sum()) # normalize the query
        if self._metric == 'angular':
            retrieved_dists = [ -1.*d/amp for d in retreived_dists ]
        else:
            retrieved_dists = [ np.sqrt(d+amp**2) for d in retreived_dists ]

        return indices[:cutoff], retrieved_dists

