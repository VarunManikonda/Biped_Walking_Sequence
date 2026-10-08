import os
import mujoco

print("🤖 Loading MuJoCo Native XML Physics Engine...")
xml_path = "biped_5dof.xml"

if not os.path.exists(xml_path):
    print(f"❌ Error: {xml_path} not found!")
else:
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        print("✅ Success! MuJoCo successfully loaded your native 5+5 DoF structural architecture.")
        print(f"   -> Detected Total Moving Joints: {model.nq - 7}") # Subtract 7 base coordinates
        print(f"   -> Total Actuators/Motors Map: {model.nu}")
    except Exception as e:
        print(f"❌ MuJoCo Compiler Error: {e}")
