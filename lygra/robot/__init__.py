# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.allegro import Allegro
from lygra.robot.shadow import Shadow
from lygra.robot.leap import Leap
from lygra.robot.dclaw import DClaw
from lygra.robot.hsl_leap import HSLLeap


def scale_canonical_space(robot, scale=1.0):
    """
    The robot's canonical box, dilated about its own centre by `scale`.

    `scale=1.0` returns the box verbatim, so this is safe to call unconditionally.
    Used to widen the sampling region for eval datasets without editing the robot
    config that the training datasets were built against.
    """
    import numpy as np

    bmin, bmax = robot.get_canonical_space()
    if bmin is None or scale == 1.0:
        return bmin, bmax

    bmin, bmax = np.asarray(bmin), np.asarray(bmax)
    centre = (bmin + bmax) / 2.0
    half = (bmax - bmin) / 2.0 * scale
    return (centre - half).astype(bmin.dtype), (centre + half).astype(bmax.dtype)


def build_robot(name, urdf_path=None):
    if name == 'allegro':
        return Allegro(urdf_path=urdf_path)
    
    elif name == 'leap':
        return Leap(urdf_path=urdf_path)

    elif name == 'hsl_leap':
        return HSLLeap(urdf_path=urdf_path)

    elif name == 'shadow':
        return Shadow(urdf_path=urdf_path)

    elif name == 'dclaw':
        return DClaw(urdf_path=urdf_path)
    
    else:
        assert False, f"Robot {name} undefined."
