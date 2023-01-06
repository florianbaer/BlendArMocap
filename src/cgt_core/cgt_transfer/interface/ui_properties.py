'''
Copyright (C) cgtinker, cgtinker.com, hello@cgtinker.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import bpy
from bpy.props import StringProperty, EnumProperty, IntProperty, BoolProperty, FloatVectorProperty, PointerProperty
from bpy.types import PropertyGroup


class CGTProperties(PropertyGroup):
    # region USER INTERFACE
    detection_input_type: EnumProperty(
        name="Type",
        description="Select detection type for motion tracking.",
        items=(
            ("stream", "Stream", ""),
            ("movie", "Movie", ""),
        )
    )

    webcam_input_device: IntProperty(
        name="Webcam Device Slot",
        description="Select Webcam device.",
        min=0,
        max=4,
        default=0
    )

    key_frame_step: IntProperty(
        name="Key Step",
        description="Select keyframe step rate.",
        min=1,
        max=12,
        default=4
    )
    # region DETECTION
    modal_active: BoolProperty(
        name="detection operator bool",
        description="helper bool to en- and disable detection operator",
        default=False
    )

    connection_operator_running: BoolProperty(
        name="connection operator bool",
        description="helper bool to ensure connection to server status",
        default=False
    )


    # region MOVIE
    mov_data_path: StringProperty(
        name="File Path",
        description="File path to .mov file.",
        default='*.mov;*mp4',
        options={'HIDDEN'},
        maxlen=1024,
        subtype='FILE_PATH'
    )

    freemocap_session_path: StringProperty(
        name="Freemocap Session Path",
        description="path to 'freemocap' session folder",
        default=r"/Users/Scylla/Downloads/sesh_2022-09-19_16_16_50_in_class_jsm/",
        options={'HIDDEN'},
        maxlen=1024,
        subtype='DIR_PATH'
    )
    # endregion
    # endregion

    # region TRANSFER
    button_transfer_animation: StringProperty(
        name="",
        description="Armature as target for detected results.",
        default="Transfer Animation"
    )

    legacy_features_bool: BoolProperty(
        name="Legacy Features",
        description="Enable legacy features which require external dependencies",
    )

    experimental_feature_bool: BoolProperty(
        name="Transfer Legs",
        description="Transfer pose legs motion to rigify rig",
        default=True
    )

    static_hands_bool: BoolProperty(
        name="Static Wrists",
        description="Transfer finger movements without wrist movement",
        default=False
    )

    overwrite_drivers_bool: BoolProperty(
        name="Overwrite Drivers",
        description="Overwrites drivers when reimporting",
        default=False
    )

    def is_rigify_armature(self, object):
        if object.type == 'ARMATURE':
            if 'rig_id' in object.data:
                return True
        return False

    def is_armature(self, object):
        if object.type == 'ARMATURE':
            if 'rig_id' in object.data:
                return False
            return True
        return False

    selected_rig: bpy.props.PointerProperty(
        type=bpy.types.Object,
        description="Select an armature for animation transfer.",
        name="Armature",
        poll=is_rigify_armature)

    selected_metarig: bpy.props.PointerProperty(
        type=bpy.types.Object,
        description="Select a metarig as future gamerig.",
        name="Armature",
        poll=is_armature)

    # TODO: USE DEFAULTS
    def cgt_collection_poll(self, object):
        return object.name.startswith('cgt_')
        # return object.name in ["cgt_FACE", "cgt_HAND", "cgt_POSE", "cgt_DRIVERS"]

    selected_driver_collection: bpy.props.PointerProperty(
        name="",
        type=bpy.types.Collection,
        description="Select a collection of Divers.",
        poll=cgt_collection_poll
    )
    # endregion

    # region SELECTION
    # ("HOLISTIC", "Holistic", ""),
    enum_detection_type: EnumProperty(
        name="Target",
        description="Select detection type for motion tracking.",
        items=(
            ("HAND", "Hands", ""),
            ("FACE", "Face", ""),
            ("POSE", "Pose", ""),
            ("HOLISTIC", "Holistic", ""),
        )
    )
    # endregion
    # endregion

    # region PREFERENCES
    enum_stream_dim: EnumProperty(
        name="Stream Dimensions",
        description="Dimensions for video Stream input.",
        items=(
            ("sd", "720x480 - recommended", ""),
            ("hd", "1240x720 - experimental", ""),
            ("fhd", "1920x1080 - experimental", ""),
        )
    )

    enum_stream_type: EnumProperty(
        name="Stream Backend",
        description="Sets Stream backend.",
        items=(
            ("0", "automatic", ""),
            ("1", "capdshow", "")
        )
    )

    transfer_type_path: StringProperty(
        name="Dir Path",
        description="Path to folder containing Hand, Pose and Face jsons.",
        default="",
        maxlen=1024,
        subtype='DIR_PATH'
    )
    # endregion

    # region REMAPPING
    toggle_drivers_bool: BoolProperty(
        name="Toggle Drivers",
        description="helper bool to en- and disable drivers",
        default=True
    )
    # endregion


def register():
    bpy.types.Scene.m_cgtinker_mediapipe = PointerProperty(type=CGTProperties)
