# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.base import RobotInterface
import numpy as np


class HSLLeap(RobotInterface):
    """
    LEAP hand variant in `my_assets/hand/hsl_leap`.

    Differences from `Leap` (./assets/hand/leap) that matter for this config:

    1. Root link.  This URDF has no separate `base` link -- `palm` *is* the root.
       That makes `palm` usable as a static contact link (see below), which is
       not the case for the stock LEAP where the palm hangs off `base`.

    2. Frame convention.  Grasps are emitted in this `palm` frame, so quantities
       copied from the stock LEAP config had to be remapped. The two hands are
       related by an exact rigid transform (verified to 0.1 mm over random
       configurations, on every shared link):

           p_palm = R @ p_base + t,   R = [[0, 0, 1],      t = [-0.04041,
                                           [0, -1, 0],           0.00466,
                                           [1, 0,  0]]           0.01128]

       i.e. fingers extend along +X (stock: +Z), curl toward +Z (stock: +X, the
       palm normal), and index -> ring runs +Y (stock: -Y). Link *orientations*
       are shared, so per-link normal rules transfer without remapping.

    3. Naming.  if/mf/rf/th (index/middle/ring/thumb) with bs/px/md/ds/tip
       segments, instead of mcp_joint/pip/dip/fingertip. The correspondence is
       one segment off from what the names suggest: stock `fingertip` is this
       hand's `*_ds`, and `*_tip` is an extra link carrying the rubber tip cap.
    """

    # The stock LEAP config points `static_link` at "base_link", which does not
    # exist in that URDF, so palm contacts are silently off there. Here `palm`
    # is the root so they actually work. Set to False for stock-LEAP parity.
    USE_PALM_CONTACT = True

    def get_canonical_space(self):
        """
        Region (in the `palm` frame) that object surface points get dragged into.

        Hand-tuned with `viz_contact_field.py`. For reference, the stock LEAP box
        [0.08, -0.03, 0.06] .. [0.14, 0.03, 0.13] pushed through the exact
        transform above lands at [0.0196, -0.0253, 0.0913] .. [0.0896, 0.0347,
        0.1513]; this box is deliberately lower and wider in z.
        """
        box_min = np.array([-0.0, -0.035, 0.047], dtype=np.float32)
        box_max = np.array([0.1, 0.035, 0.145], dtype=np.float32)
        return box_min, box_max

    def get_default_urdf_path(self):
        return './my_assets/hand/hsl_leap/urdf/leap_hand_right.urdf'

    def get_contact_field_config(self):
        """
        Which link surfaces may carry contacts, and which parts of them.

        Both rules are half-space tests on the *surface normal expressed in the
        link's own frame*, evaluated per surface patch:

          movable_link / disabled_normal: [(n, theta), ...]
              Drop the patch if any of its key vectors lies within `theta` of
              `n`. theta = pi/2 therefore deletes the whole hemisphere facing
              `n` and keeps the opposite one.

          static_link / allowed_normal: [(N, theta), ...]
              Inverse sense -- keep the patch only if its centroid normal lies
              within `theta` of one of the rows of `N`. Static patches are taken
              as already being in the hand base frame, which is only true when
              the link *is* the root; `palm` is, here.
        """
        config = {
            "type": "v1",
            "movable_link": {},
            "static_link": {}
        }

        # Fingertip links carry the rubber tip mesh as their collision geometry.
        # In the tip frame, -Y is distal and Z is the distal joint axis, so this
        # cone drops the patches facing back up the finger and off to one side,
        # keeping the distal pad. (Stock LEAP instead cuts the whole -Z
        # hemisphere, which is a purely lateral split.)
        for link in ["if_tip", "mf_tip", "rf_tip", "th_tip"]:
            config["movable_link"][link] = {
                "disabled_normal": [
                    (np.array([1.0, 1.0, 0.0]), 1.2)
                ]
            }

        # Palm. Fingers curl toward +Z, so +Z is the grasping face of the palm.
        if self.USE_PALM_CONTACT:
            config["static_link"]["palm"] = {
                "allowed_normal": [
                    (np.array([[0.0, 0.0, 1.0]]), 3.1415926 * 0.25)
                ]
            }

        return config

    def get_active_joints(self):
        """
        DOF order of the generated `q` vectors -- URDF chain order, i.e.
        [mcp, rot, pip, dip] per finger, index/middle/ring/thumb.

        Note this differs from stock LEAP, which numbers the side joint first
        ([rot, mcp, pip, dip]), so `q` here is NOT index-compatible with datasets
        generated for `leap`: entries 0/1, 4/5, 8/9 are swapped relative to those.
        """
        return [
            "if_mcp", "if_rot", "if_pip", "if_dip",
            "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
            "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
            "th_cmc", "th_axl", "th_mcp", "th_ipl",
        ]

    def get_base_link(self):
        return "palm"

    def get_static_links(self):
        return ["palm"]

    def get_mesh_scale(self):
        return 1.0
