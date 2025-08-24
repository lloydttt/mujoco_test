import mujoco
from mujoco import MjModel, MjData, Renderer
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import os
import mediapy as media
# os.environ["MUJOCO_GL"] = "egl"


if __name__ == "__main__":
    print("Mujoco test begin!")
    xml_path = os.path.join(os.path.dirname(__file__), "MJCF.xml")
    model = MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    height = 480
    width = 480
    # with mujoco.Renderer(model, height, width) as r:
    #     mujoco.mj_forward(model, data)
    #     r.update_scene(data, "fixed")

        # media.show_image(r.render())
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
        

