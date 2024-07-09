# from geotrans code
# https://github.com/qinzheng93/GeoTransformer
from .basedon_other.geotransformer_code.registratoin_result_evaluation import (
    compute_registration_rmse,
    compute_registration_error,
    compute_inlier_num_ratio
)
from .basedon_other.geotransformer_code.point_cloud_process import apply_transform



class PCRResultEvaluator:
    def __init__(self,
                 threshold_rre, threshold_rte, positive_radius
                 ):
        # threshold
        self.threshold_rre = threshold_rre
        self.threshold_rte = threshold_rte
        self.positive_radius = positive_radius

    def update_pair(self, pts_src, pts_dst, trans_gt):
        # source
        self.pts_src = pts_src
        self.pts_dst = pts_dst
        self.trans_gt = trans_gt

    def eval_corr(self, kpts_src, kpts_dst, is_print=False, kpts_src_cons=None, kpts_dst_cons=None, positive_radius=None, is_return_indices=False, prefix_info=""):
        """
        return inlier and outlier  or indic of inlier
        """
        if positive_radius is None:
            positive_radius = self.positive_radius
        # inlier_numer inlier_ratio, inlier_indice
        IN, IP, inlier_indice = compute_inlier_num_ratio(
            kpts_dst, kpts_src, self.trans_gt, positive_radius=positive_radius, is_return_indices=True)

        if is_print:
            print("{}IN: {}, IP: {:.2f}%".format(
                prefix_info,
                IN, IP*100))

        if is_return_indices:
            return IN, IP, inlier_indice
        return IN, IP

    def eval_trans(self, trans):
        """
        calculate error 
        """
        # rmse
        rmse = compute_registration_rmse(self.pts_src, self.trans_gt, trans)

        # relative transform error
        rre, rte = compute_registration_error(self.trans_gt, trans)

        # is success
        is_success = (rre <= self.threshold_rre) and (
            rte <= self.threshold_rte)

        return rmse, rre, rte, is_success