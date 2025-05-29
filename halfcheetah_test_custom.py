import os
import mujoco
import imageio
import numpy as np

# === Load custom model ===
xml_path = os.path.join(os.path.dirname(__file__), "half_cheetah.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# === Modify leg length dynamically ===
def change_leg_length(thigh_len=0.25, shin_len=0.25):
    # Get geom IDs
    bthigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bthigh")
    bshin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bshin")
    fthigh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fthigh")
    fshin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fshin")

    # Set new half-lengths (axis=1 is length for capsule)
    model.geom_size[bthigh][1] = thigh_len
    model.geom_size[fthigh][1] = thigh_len
    model.geom_size[bshin][1] = shin_len
    model.geom_size[fshin][1] = shin_len

# ✅ Apply the length change
change_leg_length(thigh_len=0.27, shin_len=0.22)

# === Camera adjustment ===
renderer = mujoco.Renderer(model, width=500, height=480)
frames = []

# === Run the simulation ===
for _ in range(200):
    mujoco.mj_step(model, data)
    renderer.update_scene(data, camera="track")  # must exist in XML
    frame = renderer.render()
    frames.append(frame)

    # Random action to make it move
    data.ctrl[:] = np.random.uniform(-1, 1, size=model.nu)

renderer.close()

# === Save GIF ===
imageio.mimsave("halfcheetah_custom.gif", frames, fps=30)
print("✅ Saved as halfcheetah_custom.gif")
