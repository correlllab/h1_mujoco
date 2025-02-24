import mujoco
import mujoco.viewer

# initialize robot model
model = mujoco.MjModel.from_xml_path('unitree_robots/h1/scene_terrain.xml')
data = mujoco.MjData(model)

# iterate through bodies
print('H1 Robot Bodies')
print('================')
for i in range(model.nbody):
    body = model.body(i)
    print(f'''Body {i}: {body.name}
          jnt_adr: {model.body_jntadr[i]}, jnt_num: {model.body_jntnum[i]}
          dof_adr: {model.body_dofadr[i]}, dof_num: {model.body_dofnum[i]}''')

# iterate through joints
print('H1 Robot Joints')
print('================')
for i in range(model.njnt):
    joint = model.joint(i)
    print(f'''Joint {i}: {joint.name}
          qpos_id: {model.jnt_qposadr[i]}, qvel_id: {model.jnt_dofadr[i]}
          limit: {joint.range}''')
