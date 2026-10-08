with open("biped_5dof.urdf", "r") as f:
    content = f.read()

# Locate the closing robot tag and inject the 10 motor actuators before it
actuators_string = """
  <!-- TRANSMISSION ACTUATORS FOR MUJOCO -->
  <actuator>
    <motor name="L_hip_roll_motor" joint="left_hip_roll" gear="15"/>
    <motor name="L_hip_pitch_motor" joint="left_hip_pitch" gear="30"/>
    <motor name="L_knee_pitch_motor" joint="left_knee_pitch" gear="30"/>
    <motor name="L_ankle_pitch_motor" joint="left_ankle_pitch" gear="15"/>
    <motor name="L_ankle_roll_motor" joint="left_ankle_roll" gear="15"/>
    
    <motor name="R_hip_roll_motor" joint="right_hip_roll" gear="15"/>
    <motor name="R_hip_pitch_motor" joint="right_hip_pitch" gear="30"/>
    <motor name="R_knee_pitch_motor" joint="right_knee_pitch" gear="30"/>
    <motor name="R_ankle_pitch_motor" joint="right_ankle_pitch" gear="15"/>
    <motor name="R_ankle_roll_motor" joint="right_ankle_roll" gear="15"/>
  </actuator>
</robot>
"""

updated_content = content.replace("</robot>", actuators_string)

with open("biped_5dof.urdf", "w") as f:
    f.write(updated_content)

print("💪 Success! Mapped 10 dynamic motor transmissions onto your biped skeleton.")
