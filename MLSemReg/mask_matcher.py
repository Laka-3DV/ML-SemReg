import numpy as np
import taichi as ti
import taichi.math as tm
import eval_taichi

ti.init(ti.gpu)
 

def get_topx_same_consis_matrix(same_matrix, topx):
    N_dst = same_matrix.shape[1]
    top_l2 = np.max(same_matrix, axis=1).astype(np.int32) - (topx - 1)
    # top_l2 = top_l1  # NOTE: base one the grade is continue with high probbility
    top_l2_broad = np.tile(top_l2.reshape(-1, 1), reps=(1, N_dst))
    global_consis = same_matrix >= top_l2_broad
    return global_consis



@ti.kernel
def taichi_calc_same_matrix2(
    out: ti.types.ndarray(ndim=2, dtype=ti.int32), 
    gss_src: ti.types.ndarray(ndim=2, dtype=ti.uint8), 
    gss_dst: ti.types.ndarray(ndim=2, dtype=ti.uint8), 
):
    for i, j, k in ti.ndrange(gss_src.shape[0], gss_dst.shape[0], gss_src.shape[1]):
        # if gss_src[i, k] == 1 and gss_dst[j, k] == 1:
        if gss_src[i, k] and gss_dst[j, k]:  
            # out[i, j] += 1
            ti.atomic_add(out[i, j], 1)  

def get_same_matrix(gss_src, gss_dst):
    import eval_taichi
    if not eval_taichi.is_use_taich:
        same_matrix = np.sum(np.bitwise_and(gss_src[:, None, ...], gss_dst[None, ...]), axis=(-1, -2))
    else:
        gss_src_flatten = gss_src.reshape(len(gss_src), -1)
        gss_dst_flatten = gss_dst.reshape(len(gss_dst), -1)
        out = np.zeros(shape=(len(gss_src), len(gss_dst)), dtype=np.int32)
        taichi_calc_same_matrix2(out, gss_src_flatten, gss_dst_flatten)

        same_matrix = out

    return same_matrix



def get_nn_select_raw_material(desc_src, desc_dst,
                               gss_src, gss_dst,
                               topx):
    # global match
    same_matrix = get_same_matrix(gss_src, gss_dst)
    # NN match
    cross_desc_dis = np.linalg.norm(
        desc_src[:, None] - desc_dst[None, :], axis=-1)  # TODO: not all calc dis
    grades = np.exp(-cross_desc_dis)

    return same_matrix, grades


def _matcher_nn_mm(global_consis, grades):
    N_src = global_consis.shape[0]

    mask_grades = grades
    # grade set to zeros for not satisfy gss consistency
    mask_grades[np.logical_not(global_consis)] = 0

    corres = np.tile(np.arange(N_src).reshape(-1, 1), reps=(1, 2))
    corres[:, 1] = np.argmax(mask_grades, axis=1)

    return corres


@ti.kernel
def taichi_calc_grades(
    out_grades: ti.types.ndarray(ndim=2, dtype=ti.float32),   
    gss_src: ti.types.ndarray(ndim=2, dtype=ti.uint8), 
    gss_dst: ti.types.ndarray(ndim=2, dtype=ti.uint8), 
    desc_src: ti.types.ndarray(ndim=2, dtype=ti.float32), 
    desc_dst: ti.types.ndarray(ndim=2, dtype=ti.float32), 
    topx: ti.int32,
    scene_similar: ti.types.ndarray(ndim=2, dtype=ti.int32), 
    # scene_similar = ti.ndarray(dtype=ti.int32, shape=(n_src, n_dst))   
    row_max: ti.types.ndarray(ndim=1, dtype=ti.int32) 
    # row_max = ti.ndarray(dtype=ti.int32, shape=(gss_src.shape[0]))
):
    n_src = gss_src.shape[0]
    n_dst = gss_dst.shape[0]
    # scene_similar = ti.field(dtype=ti.int32, shape=(n_src, n_dst))   
    for i, j, k in ti.ndrange(gss_src.shape[0], gss_dst.shape[0], gss_src.shape[1]):
        # print("?", end="")
        if gss_src[i, k] == 1 and gss_dst[j, k] == 1:
            scene_similar[i, j] += 1

    for i in range(n_src):  # para
        for j in range(n_dst):  # sera
            if row_max[i] < scene_similar[i, j]:
                row_max[i] = scene_similar[i, j] - (topx - 1)

    ti.loop_config(serialize=False) # Disable auto-parallelism in Taichi
    for I in ti.grouped(out_grades):
        out_grades[I] = 0
        # if row_max[i] >= 1 and scene_similar[i, j] >= row_max[i]:
        if scene_similar[I] >= row_max[I[0]]:
            # diss of: desc i desc j
            for k in range(desc_src.shape[1]):  
                out_grades[I] += (desc_src[I[0], k] - desc_dst[I[1], k]) * (desc_src[I[0], k] - desc_dst[I[1], k])
            out_grades[I] = ti.math.exp(-ti.math.sqrt(out_grades[I]))
        
 

def mask_match_taichi(
    desc_src, desc_dst,
    gss_src, gss_dst,
    topx
):
    # return out
    gss_src_flatten = gss_src.reshape(len(gss_src), -1)
    gss_dst_flatten = gss_dst.reshape(len(gss_dst), -1)

    n_src, n_dst = len(gss_src), len(gss_dst)

    scene_similar = np.zeros(shape=(n_src, n_dst), dtype=np.int32)
    row_max = np.zeros(shape=(n_src, ), dtype=np.int32)

    out_grades = np.zeros(shape=(len(gss_src), len(gss_dst)), dtype=np.float32)

    taichi_calc_grades(
        out_grades, 
        gss_src_flatten, 
        gss_dst_flatten, 
        desc_src.astype(np.float32), 
        desc_dst.astype(np.float32), 
        topx=topx, 
        scene_similar=scene_similar, 
        row_max=row_max
    )

    corres = np.tile(np.arange(n_src).reshape(-1, 1), reps=(1, 2))
    corres[:, 1] = np.argmax(out_grades, axis=1)

    return corres


def maskmatcher_nn(
    desc_src, desc_dst,
    gss_src, gss_dst,
    topx,
    **kwargs
):
    # global match
    if not eval_taichi.is_use_taich:
        same_matrix, grades = get_nn_select_raw_material(desc_src, desc_dst,
                                                        gss_src, gss_dst,
                                                        topx)

        global_consis = get_topx_same_consis_matrix(same_matrix, topx=topx)
        corres = _matcher_nn_mm(global_consis, grades)
        return corres
    else:
        corres2 = mask_match_taichi(
            desc_src, desc_dst,
            gss_src, gss_dst,
            topx
        )
        return corres2

