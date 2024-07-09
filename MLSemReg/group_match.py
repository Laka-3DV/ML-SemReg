import numpy as np
from sklearn.neighbors import KDTree


def query_local_signature(
        pts, label, kpts,
        radiu):
    """
    output: list of ndarray (int) [ [0, 1], [0] ] 
    """
    kdtree = KDTree(pts)
    rslt = kdtree.query_radius(kpts, r=radiu)
    lsig = [np.unique(label[lsig_idx]) for lsig_idx in rslt]
    return lsig


def create_group_labels(
    lss_list,
    label_unuse,  # just eliminate
    label_undfine  # label to every group (KITTI: 0)
):
    """
    create group label for each kpts
    Input
        kpts: ndarray Nx3
        lsigs: list of ndarray(semantic label set for this kpts)
    Output
        groups of kpts (global idx)
            e.g: {"1_34_4": [2, 33, 45] ,....}, 
                which "1_34_4" is ths processed lsig and value is ths idx of kpts in global 
    """
    lss_groups = dict()

    # query each keypoint's label and group it
    for i, lsig in enumerate(lss_list):
        if len(lsig) == 0:
            continue

        if np.intersect1d(lsig, label_unuse).size == len(lsig):
            continue

        for l in lsig:
            if l not in lss_groups.keys() and l != 0:
                lss_groups[l] = []  # init to list

            if l in label_undfine:    # undefine to every group
                for key in lss_groups.keys():
                    lss_groups[key].append(i)
            else:  # to l group
                lss_groups[l].append(i)

    return lss_groups


class GroupIter:
    def __init__(self, lsig_src, lsig_dst, label_unuse, label_undfine) -> None:
        self.lsig_src = lsig_src
        self.lsig_dst = lsig_dst

        # groups
        self.groups_src = create_group_labels(
            self.lsig_src, label_unuse, label_undfine)
        self.groups_dst = create_group_labels(
            self.lsig_dst, label_unuse, label_undfine)

        # para
        self.keys = list(self.groups_src.keys())
        self.index = 0
        self.N_lsig_src = len(self.keys)

    def __iter__(self):
        return self

    def __next__(self):
        for idx in range(self.index, self.N_lsig_src):
            key = self.keys[idx]
            if key in self.groups_dst:
                self.index = idx + 1
                return (
                    np.asarray(self.groups_src[key]),  # idx_src_sub
                    np.asarray(self.groups_dst[key])  # idx_dst_sub
                )

        raise StopIteration


class GroupingMatcher:
    def __init__(self, lsig_src, lsig_dst, config_lsig) -> None:
        self.lsig_src, self.lsig_dst = lsig_src, lsig_dst
        self.label_unuse = config_lsig.label_unuse
        self.label_undefine = config_lsig.label_undefine

        self.refresh()

    def refresh(self):
        self.group_iter = GroupIter(
            self.lsig_src, self.lsig_dst, self.label_unuse, self.label_undefine)

    def run_match_base_desc(self, match_func, desc_src, desc_dst, **kwargs):
        corres = []
        for idx_src, idx_dst in self.group_iter:
            desc_src_sub = desc_src[idx_src]
            desc_dst_sub = desc_dst[idx_dst]

            # match
            corres_sub = match_func(desc_src_sub, desc_dst_sub, **kwargs)
            corres_sub[:, 0] = idx_src[corres_sub[:, 0]]
            corres_sub[:, 1] = idx_dst[corres_sub[:, 1]]

            corres.append(corres_sub)

        self.refresh()
        corres = np.vstack(corres)
        return np.unique(corres, axis=0)

    def run_match_based_on_gss(self, match_func, desc_src, desc_dst, gss_src, gss_dst, **kwargs):
        corres = []
        for idx_src, idx_dst in self.group_iter:
            # extract data
            desc_src_sub = desc_src[idx_src]
            desc_dst_sub = desc_dst[idx_dst]

            gss_src_sub = gss_src[idx_src]
            gss_dst_sub = gss_dst[idx_dst]

            # match
            corres_sub = match_func(
                desc_src_sub, desc_dst_sub, gss_src_sub, gss_dst_sub, **kwargs)

            if len(corres_sub) > 0:
                corres_sub[:, 0] = idx_src[corres_sub[:, 0]]
                corres_sub[:, 1] = idx_dst[corres_sub[:, 1]]

                corres.append(corres_sub)
        self.refresh()
        corres = np.vstack(corres)
        return np.unique(corres, axis=0)
