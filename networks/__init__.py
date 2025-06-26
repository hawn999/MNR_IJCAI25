from .resnet4b import resnet4b
from .predrnet_v3 import predrnet_raven, predrnet_analogy, predrnet_mnr, hcvarr, scar, pred, mm, mrnet
# from .predrnet import predrnet_raven, predrnet_analogy, predrnet_mnr, hcvarr, scar, pred, mm, mrnet, \
    # mrnet_price_analogy, mrnet_pric_raven, hcv_pric_analogy, hcv_pric_raven
from .HCVARR_RPV import hcvarr_rpv
from .hpai import hpai_raven
from .hpai_pric import hpai_pric_raven, hpai_pric_analogy
from .predrnet_original_source_code import predrnet_original_raven
from .hcv_pric_v2 import hcv_pric_v2_analogy
from .DARR import darr_raven, darr_analogy

model_dict = {
    "resnet4b": resnet4b,
    "predrnet_raven": predrnet_raven,
    "predrnet_analogy": predrnet_analogy,
    "hcvarr": hcvarr,
    "scar": scar,
    "pred": pred,
    "mm": mm,
    "mrnet": mrnet,
    "predrnet_mnr": predrnet_mnr,
    # "mrnet_pric_analogy": mrnet_price_analogy,
    # "mrnet_pric_raven": mrnet_pric_raven,
    # "hcv_pric_analogy": hcv_pric_analogy,
    # "hcv_pric_raven": hcv_pric_raven,
    # "hcvarr_rpv": hcvarr_rpv,
    # "hpai_raven": hpai_raven,
    # "hpai_pric_raven": hpai_pric_raven,
    # "hpai_pric_analogy": hpai_pric_analogy,
    # "predrnet_original_raven": predrnet_original_raven,
    # "hcv_pric_v2_analogy": hcv_pric_v2_analogy,
    # "darr_raven": darr_raven,
    # "darr_analogy": darr_analogy,
}



def create_net(args):
    net = None

    kwargs = {}
    kwargs["block_drop"] = args.block_drop
    kwargs["classifier_drop"] = args.classifier_drop
    kwargs["classifier_hidreduce"] = args.classifier_hidreduce
    kwargs["num_filters"] = args.num_filters
    kwargs["num_extra_stages"] = args.num_extra_stages
    kwargs["in_channels"] = args.in_channels

    net = model_dict[args.arch.lower()](**kwargs)

    return net

