from ..encoders import CNNEncoder, PretrainedEncoder
from ..heads import BaseQHead, DuelingQHead
from ..preprocessors import BasePreprocessor, DrQPreprocessor


# Registry for modular component lookup
MODULE_REGISTRY = {
    "encoder": {
        "cnn": CNNEncoder,
        "pretrained": PretrainedEncoder,
    },
    "preprocessor": {
        "base": BasePreprocessor,
        "drq": DrQPreprocessor,
    },
    "head": {
        "base": BaseQHead,
        "dueling": DuelingQHead,
    },
}


def get_class(category: str, name: str, default: str):
    """
    Retrieve a model component class from the registry.

    Parameters
    ----------
    category : str
        Component category ("encoder", "preprocessor", or "head").
    name : str
        Desired component name (e.g. "cnn", "dueling").
    default : str
        Default component name if `name` not found.

    Returns
    -------
    type[nn.Module]
        The class corresponding to the requested component.
    """
    category = category.lower()
    name = name.lower()
    try:
        return MODULE_REGISTRY[category].get(name, MODULE_REGISTRY[category][default])
    except KeyError as e:
        raise ValueError(f"Unknown category '{category}' in module registry.") from e
