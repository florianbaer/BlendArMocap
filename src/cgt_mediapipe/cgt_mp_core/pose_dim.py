from pose_format.pose import Pose


def restore_dimensions(pose: Pose, width: float, height: float) -> Pose:
    import numpy as np
    pose.body.data = pose.body.data * np.array([width, height, 1.0])
    return pose

def restore_dimensions_div(pose: Pose, width: float, height: float) -> Pose:
    import numpy as np
    pose.body.data = pose.body.data / np.array([width, height, 1.0])
    return pose
