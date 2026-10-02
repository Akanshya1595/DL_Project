"""Maps a model tag to its builder, so shared scripts can rebuild any trained model."""
import importlib

REGISTRY = {
    "mlp":      ("model1_mlp.model", "build"),
    "cnn":      ("model2_cnn.model", "build"),
    "resnet50": ("model3_resnet50.model", "build"),
    "vit":      ("model4_vit.model", "build_vit"),
    "swin":     ("model4_vit.model", "build_swin"),
}


def get(tag):
    """tag may carry a suffix such as 'cnn_noaug'. Returns (build_fn, img_size)."""
    base = tag.replace("_noaug", "")
    mod_name, fn = REGISTRY[base]
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn), mod.IMG_SIZE
