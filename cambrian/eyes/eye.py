"""Defines the `MjCambrianEye` class, which is used to define an eye for the cambrian
environment. The eye is essentially a camera that is attached to a body in the
environment. The eye can render images and provide observations to the agent."""

from typing import Callable, List, Optional, Self, Tuple
from xml.etree.ElementTree import Element

import mujoco as mj
import numpy as np
import torch
from gymnasium import spaces
from hydra_config import HydraContainerConfig, config_wrapper
from scipy.spatial.transform import Rotation as R

from cambrian.renderer import MjCambrianRenderer, MjCambrianRendererConfig
from cambrian.renderer.overlays import MjCambrianCursor, MjCambrianViewerOverlay
from cambrian.renderer.render_utils import (
    convert_depth_distances,
    convert_depth_to_rgb,
    generate_composite,
)
from cambrian.utils import MjCambrianGeometry, device, get_logger
from cambrian.utils.cambrian_xml import MjCambrianXML
from cambrian.utils.spec import MjCambrianSpec
from cambrian.utils.types import ObsType


@config_wrapper
class MjCambrianEyeConfig(HydraContainerConfig):
    """Defines the config for an eye. Used for type hinting.

    Attributes:
        instance (Callable[[Self, str], MjCambrianEye]): The class instance to use
            when creating the eye. Takes the config and the name of the eye as
            arguments.

        fov (Tuple[float, float]): Independent of the `fovy` field in the MJCF
            xml. Used to calculate the sensorsize field. Specified in degrees. Mutually
            exclusive with `fovy`. If `focal` is unset, it is set to 1, 1. Will override
            `sensorsize`, if set. Fmt: fovy fovx.
        focal (Tuple[float, float]): The focal length of the camera.
            Fmt: focal_y focal_x.
        sensorsize (Tuple[float, float]): The size of the sensor. Fmt: height width.
        resolution (Tuple[int, int]): The width and height of the rendered image.
            Fmt: height width.
        coord (Tuple[float, float]): The x and y coordinates of the eye.
            This is used to determine the placement of the eye on the agent.
            Specified in degrees. This attr isn't actually used by eye, but by the
            agent. The eye has no knowledge of the geometry it's trying to be placed
            on. Fmt: lat lon
        gaze_coord (Optional[Tuple[float, float]]): Optional aim coordinate for the
            optical axis, specified as lat lon in degrees. If unset, the eye aims along
            `coord`, preserving the historical behavior where placement and gaze are
            coupled.
        orthographic (bool): Whether the camera is orthographic

        noise_std (float): Standard deviation of the Gaussian noise to be added to
            the rendered image. If 0, no noise is applied.
        integration_factor (float): Factor in [0, 1] controlling first-order temporal
            integration. Higher values retain more of the previous observation. The
            update is deliberately independent of image motion so the sensor model does
            not build the desired fixation behavior into the observation function.

        actuated (bool): Whether the eye is mounted on a 2-DOF (pan/tilt) gimbal with
            position-servo actuators. If True, `generate_xml` builds a small massless
            gimbal body between the parent body and the camera, adding two hinge joints
            (pan/azimuth, tilt/elevation) and two position actuators. If False
            (default), the camera is rigidly welded to the parent body as before, so
            every existing task/agent is unchanged.
        pan_range (Tuple[float, float]): The pan (azimuth) joint range in DEGREES.
            Only used when `actuated` is True.
        tilt_range (Tuple[float, float]): The tilt (elevation) joint range in DEGREES.
            Only used when `actuated` is True.
        actuator_kp (float): The position-servo stiffness (kp) for the pan/tilt
            actuators. Only used when `actuated` is True.
        joint_damping (float): The damping applied to the pan/tilt hinge joints.
            Only used when `actuated` is True.

        renderer (MjCambrianRendererConfig): The renderer config to use for the
            underlying renderer.
    """

    instance: Callable[[Self, str], "MjCambrianEye"]

    fov: Tuple[float, float]
    focal: Tuple[float, float]
    sensorsize: Tuple[float, float]
    resolution: Tuple[int, int]
    coord: Tuple[float, float]
    gaze_coord: Optional[Tuple[float, float]]
    orthographic: bool

    noise_std: float
    integration_factor: float

    actuated: bool
    pan_range: Tuple[float, float]
    tilt_range: Tuple[float, float]
    actuator_kp: float
    joint_damping: float

    renderer: MjCambrianRendererConfig


class MjCambrianEye:
    """Defines an eye for the cambrian environment. It essentially wraps a mujoco Camera
    object and provides some helper methods for rendering and generating the XML. The
    eye is attached to the parent body such that movement of the parent body will move
    the eye.

    Args:
        config (MjCambrianEyeConfig): The configuration for the eye.
        name (str): The name of the eye.

    Keyword Args:
        disable_render (bool): Whether to disable rendering. Defaults to False.
            This is useful for derived classes which don't intend to use the default
            rendering mechanism.
    """

    def __init__(
        self, config: MjCambrianEyeConfig, name: str, *, disable_render: bool = False
    ):
        self._config = config
        self._name = name

        if not 0.0 <= self._config.integration_factor <= 1.0:
            raise ValueError(
                "integration_factor must be in [0, 1], got "
                f"{self._config.integration_factor}"
            )
        if self._config.noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {self._config.noise_std}")

        self._renders_rgb = "rgb_array" in self._config.renderer.render_modes
        self._renders_depth = "depth_array" in self._config.renderer.render_modes
        assert (
            self._renders_rgb or self._renders_depth
        ), "Need at least one render mode."

        self._prev_obs_shape = self.observation_space.shape
        self._prev_obs: torch.Tensor = None
        self._fixedcamid = -1
        self._spec: MjCambrianSpec = None

        self._has_prev_obs = False

        self._renderer: MjCambrianRenderer = None
        if not disable_render:
            self._renderer = MjCambrianRenderer(self._config.renderer)

    def generate_xml(
        self,
        parent_xml: MjCambrianXML,
        geom: MjCambrianGeometry,
        parent_body_name: Optional[str] = None,
        parent: Optional[List[Element] | Element] = None,
    ) -> MjCambrianXML:
        """Generate the xml for the eye.

        In order to combine the xml for an eye with the xml for the agent that it's
        attached to, we need to replicate the path with which we want to attach the eye.
        For instance, if the body with which we want to attach the eye to is at
        `mujoco/worldbody/torso`, then we need to replicate that path in the new xml.
        This is kind of difficult with the `xml` library, but we'll utilize the
        `CambrianXML` helpers for this.

        Args:
            parent_xml (MjCambrianXML): The xml of the parent body. Used as a reference
                to extract the path of the parent body.
            geom (MjCambrianGeometry): The geometry of the parent body. Used to
                calculate the pos and quat of the eye.
            parent_body_name (Optional[str]): The name of the parent body. Will
                search for the body tag with this name, i.e.
                <body name="<parent_body_name>" ...>. Either this or `parent` must be
                set.
            parent (Optional[List[Element] | Element]): The parent element to attach
                the eye to. If set, `parent_body_name` will be ignored. Either this or
                `parent_body_name` must be set.
        """

        xml = MjCambrianXML.make_empty()

        if parent is None:
            # Get the parent body reference
            parent_body = parent_xml.find(".//body", name=parent_body_name)
            assert parent_body is not None, f"Could not find body '{parent_body_name}'."

            # Iterate through the path and add the parent elements to the new xml
            parent = None
            elements, _ = parent_xml.get_path(parent_body)
            for element in elements:
                if (
                    temp_parent := xml.find(f".//{element.tag}", **element.attrib)
                ) is not None:
                    # If the element already exists, then we'll use that as the parent
                    parent = temp_parent
                    continue
                parent = xml.add(parent, element.tag, **element.attrib)
            assert parent is not None, f"Could not find parent for '{parent_body_name}'"

        # Finally add the camera element at the end
        pos, quat = self._calculate_pos_quat(geom, self._config.coord)
        resolution = [1, 1]
        if self._renderer is not None:
            resolution = [self._renderer.config.width, self._renderer.config.height]

        camera_kwargs = dict(
            name=self._name,
            mode="fixed",
            focal=" ".join(map(str, self._config.focal)),
            sensorsize=" ".join(map(str, self._config.sensorsize)),
            resolution=" ".join(map(str, resolution)),
            orthographic=str(self._config.orthographic).lower(),
        )

        if not self._config.actuated:
            # Default behaviour: the camera is rigidly welded to the parent body. The
            # base pos/quat (aim) are baked directly onto the camera.
            xml.add(
                parent,
                "camera",
                pos=" ".join(map(str, pos)),
                quat=" ".join(map(str, quat)),
                **camera_kwargs,
            )
            return xml

        # Actuated eye: mount the camera on a 2-DOF pan/tilt gimbal. The base
        # pos/quat (aim) move to the outer mount body; the camera sits at the inner
        # link origin (identity), so with both joints at 0 the view matches the
        # welded camera exactly. Pan rotates about the mount-local up axis (+y) and
        # tilt about the link-local right axis (+x) -- verify signs empirically.
        self._add_gimbal_camera(xml, parent, pos, quat, camera_kwargs)

        return xml

    def _add_gimbal_camera(self, xml, parent, pos, quat, camera_kwargs):
        """Build the 2-DOF gimbal body tree and the pan/tilt position actuators.

        MuJoCo needs non-degenerate inertia on any body that carries a joint, so each
        gimbal body gets a tiny explicit ``<inertial>`` (the massless-body trap). Joint
        ``range`` is in degrees (compiler ``angle="degree"``) while the position-servo
        ``ctrlrange`` is in radians, matching the hinge qpos target convention used by
        the existing yaw actuator.
        """
        cfg = self._config
        # Tiny but non-degenerate inertia so the jointed bodies compile and stay stable.
        inertial = dict(pos="0 0 0", mass="1e-3", diaginertia="1e-4 1e-4 1e-4")

        # Outer (pan) body carries the base aim of the eye.
        mount = xml.add(
            parent,
            "body",
            name=f"{self._name}_mount",
            pos=" ".join(map(str, pos)),
            quat=" ".join(map(str, quat)),
        )
        xml.add(mount, "inertial", **inertial)
        xml.add(
            mount,
            "joint",
            name=f"{self._name}_pan",
            type="hinge",
            axis="0 1 0",
            range=" ".join(map(str, cfg.pan_range)),
            damping=str(cfg.joint_damping),
            limited="true",
        )

        # Inner (tilt) body holds the camera.
        link = xml.add(mount, "body", name=f"{self._name}_tilt_link", pos="0 0 0")
        xml.add(link, "inertial", **inertial)
        xml.add(
            link,
            "joint",
            name=f"{self._name}_tilt",
            type="hinge",
            axis="1 0 0",
            range=" ".join(map(str, cfg.tilt_range)),
            damping=str(cfg.joint_damping),
            limited="true",
        )
        xml.add(link, "camera", pos="0 0 0", **camera_kwargs)

        # Position servos. ctrlrange is in radians (hinge qpos target), so convert the
        # degree ranges above.
        pan_ctrl = np.deg2rad(cfg.pan_range)
        tilt_ctrl = np.deg2rad(cfg.tilt_range)
        actuator = xml.add(xml.root, "actuator")
        xml.add(
            actuator,
            "position",
            name=f"{self._name}_pan_act",
            joint=f"{self._name}_pan",
            ctrlrange=" ".join(map(str, pan_ctrl)),
            ctrllimited="true",
            kp=str(cfg.actuator_kp),
        )
        xml.add(
            actuator,
            "position",
            name=f"{self._name}_tilt_act",
            joint=f"{self._name}_tilt",
            ctrlrange=" ".join(map(str, tilt_ctrl)),
            ctrllimited="true",
            kp=str(cfg.actuator_kp),
        )

    def _calculate_pos_quat(
        self, geom: MjCambrianGeometry, coord: Tuple[float, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the position and quaternion of the eye based on the geometry of
        the parent body. The position is calculated by moving the eye to the edge of the
        geometry in the negative x direction. The quaternion is calculated by rotating
        the eye to face the center of the geometry.

        Todo:
            rotations are weird. fix this.
        """
        gaze_coord = self._config.gaze_coord
        if gaze_coord is None:
            gaze_coord = coord

        lat, lon = torch.deg2rad(torch.tensor(coord))
        gaze_lat, gaze_lon = torch.deg2rad(torch.tensor(gaze_coord))
        lon += torch.pi / 2
        gaze_lon += torch.pi / 2

        default_rot = R.from_euler("z", torch.pi / 2)
        pos_rot = default_rot * R.from_euler("yz", [lat, lon])
        rot_rot = (
            R.from_euler("z", gaze_lat)
            * R.from_euler("y", -gaze_lon)
            * default_rot
        )

        pos = pos_rot.apply([-geom.rbound, 0, 0]) + geom.pos
        quat = rot_rot.as_quat()
        return pos, quat

    def reset(self, spec: MjCambrianSpec) -> ObsType:
        """Sets up the camera for rendering. This should be called before rendering
        the first time."""

        self._spec = spec

        if self._renderer is None:
            return self.step()

        resolution = [self._renderer.config.width, self._renderer.config.height]
        self._renderer.reset(spec, *resolution)

        self._fixedcamid = spec.get_camera_id(self._name)
        assert self._fixedcamid != -1, f"Camera '{self._name}' not found."
        self._renderer.viewer.camera.type = mj.mjtCamera.mjCAMERA_FIXED
        self._renderer.viewer.camera.fixedcamid = self._fixedcamid

        self._prev_obs = torch.zeros(
            self._prev_obs_shape,
            dtype=torch.float32,
            device=device,
        )
        self._has_prev_obs = False

        obs = self.step()
        if obs.device != self._prev_obs.device:
            get_logger().warning(
                "Device mismatch. obs.device: "
                f"{obs.device}, self._prev_obs.device: {self._prev_obs.device}"
            )
        return obs

    def step(self, obs: ObsType = None) -> ObsType:
        """Simply calls `render` and sets the last observation. See `render()` for more
        information.

        Args:
            obs (Optional[ObsType]): The observation to set. Defaults to
                None. This can be used by derived classes to set the observation
                directly.
        """
        if obs is None:
            assert self._renderer is not None, "Cannot step without a renderer."
            obs = self._renderer.render()
            if self._renders_rgb and self._renders_depth:
                # If both are rendered, then we only return the rgb
                get_logger().warning(
                    "Both rgb and depth are rendered. Using only rgb.",
                    extra={"once": True},
                )
                obs = obs[0]

        obs = self._apply_sensor_noise(obs)
        obs = self._integrate_observation(obs)

        return self._update_obs(obs)

    def _update_obs(self, obs: ObsType) -> ObsType:
        """Update the observation space."""
        self._prev_obs.copy_(obs, non_blocking=True)
        self._has_prev_obs = True
        return self._prev_obs

    def _apply_sensor_noise(self, obs: ObsType) -> ObsType:
        """Add Gaussian noise to the observation if configured."""

        std = self._config.noise_std
        if std == 0.0:
            return obs

        noise = torch.normal(mean=0.0, std=std, size=obs.shape, device=obs.device)
        return torch.clamp(obs + noise, 0, 1)

    def _integrate_observation(self, obs: ObsType) -> ObsType:
        """Apply a motion-independent first-order temporal low-pass filter.

        ``integration_factor`` is the fraction of the previous sensor state retained at
        each environment step.  A value of zero is an instantaneous sensor; a value of
        one freezes the first observation.  Because the coefficient does not depend on
        frame-to-frame image change, any fixation behavior must be learned from the
        consequences of the sensor dynamics rather than being explicitly amplified by
        this function.
        """

        alpha = self._config.integration_factor

        if alpha == 0.0 or not self._has_prev_obs:
            return obs

        return (alpha * self._prev_obs) + ((1.0 - alpha) * obs)

    def render(self) -> List[MjCambrianViewerOverlay]:
        """Render the image from the camera. Will always only return the rgb array.

        This differs from step in that this is a debug method. The rendered image here
        will be used to visualize the eye in the viewer.
        """
        if self._renders_depth and not self._renders_rgb:
            image = convert_depth_to_rgb(
                convert_depth_distances(self._spec.model, self._prev_obs),
                znear=0,
                zfar=self._spec.model.stat.extent,
            )
        else:
            image = self._prev_obs

        # Composite (single-element) so the panel gets the red border and the [0, 255]
        # pixel scaling that `mjr_drawPixels` expects -- identical to the multi-eye
        # path. Passing the raw [0, 1] observation draws as near-black, which made the
        # actuated single-eye agent's POV inset appear absent in eval videos.
        lat, lon = self._config.coord
        image = generate_composite({lat: {lon: image}}) * 255.0

        position = MjCambrianCursor.Position.BOTTOM_LEFT
        layer = MjCambrianCursor.Layer.BACK
        cursor = MjCambrianCursor(position=position, x=0, y=0, layer=layer)
        return [MjCambrianViewerOverlay.create_image_overlay(image, cursor=cursor)]

    @property
    def config(self) -> MjCambrianEyeConfig:
        """The config for the eye."""
        return self._config

    @property
    def name(self) -> str:
        """The name of the eye."""
        return self._name

    @property
    def observation_space(self) -> spaces.Box:
        """Constructs the observation space for the eye. The observation space is a
        `spaces.Box` with the shape of the resolution of the eye."""

        shape = (
            (*self._config.resolution, 3)
            if self._renders_rgb
            else self._config.resolution
        )
        return spaces.Box(0.0, 1.0, shape=shape, dtype=np.float32)

    @property
    def prev_obs(self) -> torch.Tensor:
        """The last observation returned by `self.render()`."""
        return self._prev_obs
