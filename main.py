import mujoco
from mujoco import MjModel, MjData, Renderer
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import os
import mediapy as media
# os.environ["MUJOCO_GL"] = "egl"
import time


if __name__ == "__main__":
    print("Mujoco test begin!")
    xml_path = os.path.join(os.path.dirname(__file__), "MJCF.xml")
    model = MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # height = 480
    # width = 480
    # with mujoco.Renderer(model, height, width) as r:
    #     mujoco.mj_forward(model, data)
    #     r.update_scene(data, "fixed")

        # media.show_image(r.render())

    # contact force visualization
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

    # optional：调节显示效果
    model.vis.scale.contactwidth = 0.1
    model.vis.scale.contactheight = 0.03
    model.vis.scale.forcewidth = 0.05
    model.vis.map.force = 0.3
    run_speed = 1.0  # =1 实时，=0.5 半速，=2 两倍速
    dt = model.opt.timestep

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        # while viewer.is_running():
        #     mujoco.mj_step(model, data)
        #     viewer.sync()
        while viewer.is_running():
            start = time.perf_counter()
    
            mujoco.mj_step(model, data)   # 只走一步物理步长 dt
            viewer.sync()                  # 渲染一帧
    
            # 睡到下一步，保证仿真时间 ≈ 真实时间 * run_speed
            spent = time.perf_counter() - start
            remain = (dt / run_speed) - spent
            if remain > 0:
                time.sleep(remain)

