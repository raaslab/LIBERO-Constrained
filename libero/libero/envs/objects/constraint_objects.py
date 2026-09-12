import re

from robosuite.models.objects import CylinderObject

from libero.libero.envs.base_object import register_object


@register_object
class KeepOutCylinder(CylinderObject):
    """A red cylindrical marker denoting a keep-out zone: a spatial region the
    robot/end-effector must avoid while completing the task. It is placed as a
    static fixture (no free joint) so it does not add degrees of freedom to
    the simulation state.
    """

    def __init__(self, name="keep_out_cylinder", joints=None, size=(0.03, 0.09)):
        super().__init__(
            name=name,
            size=size,
            rgba=(0.8, 0.05, 0.05, 1.0),
            joints=joints,
            density=1000,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}
        self.rotation = {"z": (0, 0)}
        self.rotation_axis = "z"
