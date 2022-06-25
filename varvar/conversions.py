import json

import numpy as np


def to_lists_of_lists(tree):
    # returns either
    #   True, left stuff, right stuff, threshold, default direction, feature index
    # or
    #   False, value
    if tree[0][-1] == -1:
        return False, tree[0][1]
    return (
        True,
        to_lists_of_lists(tree[1:]),
        to_lists_of_lists(tree[int(tree[0][0]) + 1:]),
        tree[0][1], tree[0][2], tree[0][3]
    )


def back_to_lists(ll):
    # return list of tuples of either
    #   True, threshold, default direction, feature index, left child index, right child index
    # or
    #   False, value
    ret = []
    s = [ll]
    i = 0
    while i < len(s):
        has_children, *t = s[i]
        if has_children:
            ret.append((True,) + tuple(t[-3:]) + (len(s), len(s) + 1))
            s.append(t[0])
            s.append(t[1])
        else:
            ret.append((False, t[0]))
        i += 1
        
    return ret


def reorder_for_xgboost(tree):
    ll = to_lists_of_lists(tree)
    ret = back_to_lists(ll)
    return ret


def mvt_to_xgboost_dict(trees, feature_names):
    trees = list(map(reorder_for_xgboost, trees))
    
    feature_names = list(feature_names)
    base_score = 0
    num_trees = len(trees)
    
    xgb_trees = []
    missing = 0
    for i, tree in enumerate(trees):
        if len(tree) == 1 and not tree[0][0]:
            num_trees -= 1
            missing += 1
            base_score += np.log(tree[0][1])
            continue
            
        i = i - missing
            
        left_children = [
            t[-2] if has_children else -1
            for has_children, *t in tree
        ]
        right_children = [
            t[-1] if has_children else -1
            for has_children, *t in tree
        ]
        parents = {
            v: k
            for d in [left_children, right_children]
            for k, v in enumerate(d)
            if v != -1
        }
        parents = [
            parents.get(v, 2**31 - 1)
            for v in range(len(tree))
        ]
        
        xgb_trees.append({
            "categories": [],
            "categories_nodes": [],
            "categories_segments": [],
            "categories_sizes": [],
            "id": i,
            "tree_param": {
                "num_deleted": "0",
                "num_feature": str(len(feature_names)),
                "num_nodes": str(len(tree)),
                "size_leaf_vector": "0"
            },
            "split_type": [0] * len(tree),
            "split_indices": [t[2] if has_children else 0 for has_children, *t in tree],
            "default_left": [0 if has_children and t[1] == 'right' else 1 for has_children, *t in tree],
            "split_conditions": [t[0] if has_children else np.log(t[0]) for has_children, *t in tree],
            "loss_changes": [0.1] * len(tree),
            "parents": parents,
            "left_children": left_children,
            "right_children": right_children,
            "sum_hessian": [0.] * len(tree),
            "base_weights": [0.1] * len(tree),
        })
        
    xgb_trees[0]['split_conditions'] = [
        t + base_score if left_child == right_child == -1 else t
        for left_child, right_child, t in zip(
            xgb_trees[0]['left_children'], 
            xgb_trees[0]['right_children'], 
            xgb_trees[0]['split_conditions']
        )
    ]
        
    return {
        "learner": {
            "attributes": {
                "best_iteration": str(num_trees - 1),
                "best_ntree_limit": str(num_trees)
            },
            "feature_names": feature_names,
            "feature_types": ["float"] * len(feature_names),
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {
                        "num_parallel_tree": "1",
                        "num_trees": str(num_trees),
                        "size_leaf_vector": "0"
                    },
                    "tree_info": [0] * num_trees,
                    "trees": xgb_trees
                },
                "name": "gbtree"
            },
            "learner_model_param": {
                # xgboost 1.6.1 ignores "base_score",
                # so we use 0 in case one day it doesn't
                "base_score": "0", 
                "num_class": "0",
                "num_feature": str(len(feature_names)),
                "num_target": "1"
            },
            "objective": {
                "name": "reg:squarederror",
                "reg_loss_param": {
                    "scale_pos_weight": "1"
                }
            }
        },
        "version": [1, 6, 1]
    }


def dict_to_xgboost(j):
    import xgboost
    return xgboost.Booster(model_file=bytearray(json.dumps(j).encode()))


def mvt_to_xgboost(trees, feature_names):
    j = mvt_to_xgboost_dict(trees, feature_names)
    b = dict_to_xgboost(j)
    return b
