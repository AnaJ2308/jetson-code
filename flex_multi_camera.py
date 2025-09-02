# Flexible multi-camera RealSense launch (2–4 cams, cams 3/4 optional)
import copy, sys, os, yaml, pathlib
from launch import LaunchDescription, LaunchContext
import launch_ros.actions
from launch.actions import OpaqueFunction, LogInfo, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
import myrs_launch  # your helper module

# ---- User-tunable top-level args (namespaces, names, mount frame, TF poses) ----
local_parameters = [
    # Namespaces & camera names
    {'name':'camera_namespace1','default':'topic_1','description':'cam1 namespace'},
    {'name':'camera_namespace2','default':'topic_2','description':'cam2 namespace'},
    {'name':'camera_namespace3','default':'topic_3','description':'cam3 namespace'},
    {'name':'camera_namespace4','default':'topic_4','description':'cam4 namespace'},

    {'name':'camera_name1','default':'cam_1','description':'cam1 name'},
    {'name':'camera_name2','default':'cam_2','description':'cam2 name'},
    {'name':'camera_name3','default':'cam_3','description':'cam3 name'},
    {'name':'camera_name4','default':'cam_4','description':'cam4 name'},

    # Common root frame for your rig/mount
    {'name':'mount_frame','default':'base_link','description':'common root for all cameras'},

    # Static TF poses (meters, radians) relative to mount_frame
    {'name':'cam1_xyz','default':'0 0 0','description':'x y z for cam1'},
    {'name':'cam1_rpy','default':'0 0 0','description':'r p y for cam1'},

    {'name':'cam2_xyz','default':'10 0 0','description':'x y z for cam2 (spaced for visibility)'},
    {'name':'cam2_rpy','default':'0 0 0','description':'r p y for cam2'},

    {'name':'cam3_xyz','default':'20 0 0','description':'x y z for cam3'},
    {'name':'cam3_rpy','default':'0 0 0','description':'r p y for cam3'},

    {'name':'cam4_xyz','default':'30 0 0','description':'x y z for cam4'},
    {'name':'cam4_rpy','default':'0 0 0','description':'r p y for cam4'},
]

def yaml_to_dict(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def set_configurable_parameters(local_params):
    return {p['original_name']: LaunchConfiguration(p['name']) for p in local_params}

def duplicate_params(general_params, suffix: str):
    local = copy.deepcopy(general_params)
    for p in local:
        p['original_name'] = p['name']
        p['name'] += suffix
    return local

# ---- Camera launcher: only start if serial_no{idx} is provided ----
def launch_cam(context: LaunchContext, idx: int, params):
    serial = context.launch_configurations.get(f"serial_no{idx}", "")
    if not serial:
        return [LogInfo(msg=f"Skipping cam{idx}: serial_no{idx} not provided")]
    # Start RealSense via your myrs_launch helper
    return myrs_launch.launch_setup(context, params=params, param_name_suffix=str(idx))

# ---- Static TFs: create one per present camera (parent = mount_frame, child = <name>_link) ----
def launch_static_tfs(context: LaunchContext):
    # optional lifecycle support (kept minimal)
    try:
        cfg = os.path.join(os.path.dirname(__file__), '..', 'config', 'global_settings.yaml')
        use_lc = yaml_to_dict(cfg).get("use_lifecycle_node", False)
    except Exception:
        use_lc = False
    NodeAction = launch_ros.actions.LifecycleNode if use_lc else launch_ros.actions.Node

    def tf_node(i):
        serial = context.launch_configurations.get(f"serial_no{i}", "")
        if not serial:
            return None
        root = context.launch_configurations['mount_frame']
        name = context.launch_configurations[f'camera_name{i}']
        child = f"{name}_link"

        # Parse xyz/rpy strings "x y z" and "r p y"
        xyz = context.launch_configurations[f'cam{i}_xyz'].split()
        rpy = context.launch_configurations[f'cam{i}_rpy'].split()
        # Ensure length 3 each
        xyz = (xyz + ['0','0','0'])[:3]
        rpy = (rpy + ['0','0','0'])[:3]

        return NodeAction(
            package="tf2_ros", executable="static_transform_publisher",
            name=f"tf_cam{i}", namespace="",
            # x y z roll pitch yaw parent child
            arguments=[xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2], root, child]
        )

    nodes = []
    for i in (1,2,3,4):
        n = tf_node(i)
        if n: nodes.append(n)
    if not nodes:
        nodes.append(LogInfo(msg="No cameras enabled → no TFs published"))
    return nodes

def generate_launch_description():
    # Reuse the RealSense parameter set for up to 4 cameras
    params1 = duplicate_params(myrs_launch.configurable_parameters, '1')
    params2 = duplicate_params(myrs_launch.configurable_parameters, '2')
    params3 = duplicate_params(myrs_launch.configurable_parameters, '3')
    params4 = duplicate_params(myrs_launch.configurable_parameters, '4')

    # Declarations: local (namespaces/names/mount/poses) + all RS params per camera
    declares = (
        myrs_launch.declare_configurable_parameters(local_parameters) +
        myrs_launch.declare_configurable_parameters(params1) +
        myrs_launch.declare_configurable_parameters(params2) +
        myrs_launch.declare_configurable_parameters(params3) +
        myrs_launch.declare_configurable_parameters(params4)
    )

    actions = []
    # Stagger the camera startups to avoid USB race conditions
    actions += [OpaqueFunction(function=launch_cam,
                               kwargs={'idx':1, 'params': set_configurable_parameters(params1)})]              # t=0.0s
    actions += [TimerAction(period=1.5, actions=[OpaqueFunction(function=launch_cam,
                               kwargs={'idx':2, 'params': set_configurable_parameters(params2)})])]            # t=1.5s
    actions += [TimerAction(period=3.0, actions=[OpaqueFunction(function=launch_cam,
                               kwargs={'idx':3, 'params': set_configurable_parameters(params3)})])]            # t=3.0s
    actions += [TimerAction(period=4.5, actions=[OpaqueFunction(function=launch_cam,
                               kwargs={'idx':4, 'params': set_configurable_parameters(params4)})])]            # t=4.5s

    # Publish static TFs for whichever cams are actually enabled (serial provided)
    actions += [TimerAction(period=5.0, actions=[OpaqueFunction(function=launch_static_tfs)])]

    return LaunchDescription(declares + actions)
