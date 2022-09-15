"""Version info for vtk-override.

On the ``main`` branch, use 'dev0' to denote a development version.
For example:

version_info = 0, 27, 'dev0'

---

When generating pre-release wheels, use '0rcN', for example:

version_info = 0, 28, '0rc1'

Denotes the first release candidate.

"""
from collections import namedtuple
import warnings

try:
    from vtkmodules.vtkCommonCore import vtkVersion

    VTK9 = vtkVersion().GetVTKMajorVersion() >= 9
except ImportError:  # pragma: no cover
    VTK9 = False

# major, minor, patch
version_info = 0, 0, 1

# Nice string for the version
__version__ = ".".join(map(str, version_info))


def VTKVersionInfo():
    """Return the vtk version as a namedtuple."""
    version_info = namedtuple("VTKVersionInfo", ["major", "minor", "micro"])

    try:
        ver = vtkVersion()
        major = ver.GetVTKMajorVersion()
        minor = ver.GetVTKMinorVersion()
        micro = ver.GetVTKBuildVersion()
    except NameError:  # pragma: no cover
        warnings.warn("Unable to detect VTK version. Defaulting to v4.0.0")
        major, minor, micro = (4, 0, 0)

    return version_info(major, minor, micro)
