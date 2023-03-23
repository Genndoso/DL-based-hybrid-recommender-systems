from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, diags
import numpy as np

class SVD_baseline:
    def __init__(self, train_matrix, data_description, config):
        self.train_matrix = train_matrix
        self.config = config
        self.data_description = data_description

    def rescale_matrix(self, matrix):

        frequencies = matrix.getnnz(axis=0)
        scaling_weights = np.power(frequencies, 0.5 * (self.config['scaling'] - 1))
        return matrix.dot(diags(scaling_weights)), scaling_weights

    def build_svd_model(self):
        scaled_matrix, scaling_weights = self.rescale_matrix(self.train_matrix)
        u, s, vt = svds(scaled_matrix, k=self.config['rank'])

        item_factors = vt.T  # np.ascontiguousarray(vt[::-1, :].T)
        user_factors = u  # np.ascontiguousarray(u[::-1, :])
        return user_factors, item_factors  # , singular_values


    def user_evaluate(self, user_id, preds, holdout):
        n_items = self.data_description['n_items']
        holdout_user = holdout[holdout.user_id == user_id]
        predictions = preds[user_id]
        hits_mask = predictions == holdout_user.item_id.values
        # HR calculation
        hr = np.mean(hits_mask.any(axis=0))
        # MRR calculation
        n_test_users = preds.shape[0]
        hit_rank = np.where(hits_mask)[0] + 1.0

        mrr = np.sum(1 / hit_rank)
        # coverage calculation
        cov = np.unique(preds).size / n_items
        # ndcg
        # NDCG
        ndcg_per_user = 1.0 / np.log2(hit_rank + 1)
        ndcg = np.sum(ndcg_per_user) / n_test_users

        return hr, mrr, cov, ndcg

    def svd_evaluate(self,holdout, users_to_eval = 10000, topn = 20):
        hr_full = []
        mrr_full = []
        cov_full = []
        ndcg_full = []
        user_factors, item_factors = self.build_svd_model()
        scores = csr_matrix(user_factors[:users_to_eval, :]).dot(csr_matrix(item_factors).T)
        scores_topn = topn_recommendations(scores.A, topn)

        for i in holdout.user_id.sort_values():
            if i % 1000 == 0:
                print(i)
            if i == users_to_eval:
                break
            # predictions = scores_topn[i]

            hr, mrr, cov, ndcg = self.user_evaluate(i, scores_topn, holdout)
            hr_full.append(hr)
            mrr_full.append(mrr)
            cov_full.append(cov)
            ndcg_full.append(ndcg)
        return np.array(hr_full), np.array(mrr_full), np.array(cov_full), np.array(ndcg_full)

def topn_recommendations(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations


def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]
