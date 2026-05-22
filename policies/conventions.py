# Prediction
ACTION = "action"

# Observation key
OBS_ENV_STATE = "start_state"
OBS_ROBOT_STATE = "start_pos_world"
OBS_VISUAL_STATE = "visual_feature_patch"
OBS_GRIPPER_VISUAL_STATE = "visual_feature_patch_gripper"
OBS_HEAD_CAM_STATE = "T_world_cam"
OBS_GRIPPER_CAM_STATE = "T_world_grippercam"

# History observation key
HISTORY_OBS_ENV_STATE = "history_state"
HISTORY_OBS_ROBOT_STATE = "history_action"
HISTORY_OBS_VISUAL_STATE = "history_visual_feature_patch"
HISTORY_OBS_GRIPPER_VISUAL_STATE = "history_visual_feature_patch_gripper"
HISTORY_OBS_HEAD_CAM_STATE = "history_raymap"
HISTORY_OBS_GRIPPER_CAM_STATE = "history_raymap_gripper"
