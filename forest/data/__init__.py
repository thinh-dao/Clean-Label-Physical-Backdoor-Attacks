"""Basic data handling."""
from .kettle_single import KettleSingle
__all__ = ['Kettle']

def Kettle(args, batch_size, augmentations, mixing_method, setup):
    """Implement Main interface."""
    return KettleSingle(args, batch_size, augmentations, mixing_method, setup)