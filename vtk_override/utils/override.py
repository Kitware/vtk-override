import warnings

from vtkmodules.vtkCommonCore import vtkObjectBase

WARN_ON_FAILED_OVERRIDE: bool = True


class FailedOverrideWarning(Warning):
    pass


def override(to_override: vtkObjectBase, *args):
    """Class decorator to override VTK classes.

    This is backward compatible with VTK in that it will skip the override
    if VTK<9.2, but raise a ``FailedOverrideWarning`` if
    ``WARN_ON_FAILED_OVERRIDE`` is true.

    """

    def wrapper(subclass):
        try:
            return to_override.override(subclass)
        except AttributeError:
            if WARN_ON_FAILED_OVERRIDE:
                warnings.warn(
                    FailedOverrideWarning(
                        f"Unable to override VTK class {to_override} with {subclass} "
                        "as it has no `override` method. Please make sure VTK is version 9.2 or above."
                    )
                )
        return subclass

    if len(args):
        return wrapper(*args)

    return wrapper
