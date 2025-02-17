
import numpy as np
import pyvista as pv
from pyquaternion import Quaternion

import mujoco
from robot_descriptions.loaders.mujoco import load_robot_description

def mj_get_body_mesh(model, id):
    '''
    Given the MuJoCo model and body id, return the mesh of the body.
    If the body has multiple meshes, merge them into one mesh.
    '''
    # get body
    mj_body = model.body(id)
    # inspect geoms in the body
    mj_geom_adr = mj_body.geomadr[0]
    mj_geom_num = mj_body.geomnum[0]

    # collect all the meshes
    mesh_list = []
    for i in range(mj_geom_num):
        geom = model.geom(mj_geom_adr + i)
        if geom.contype == 0 and geom.dataid != -1:
            mesh = mj_mesh_to_polydata(model, geom.dataid)
            # convert quaternion to 4x4 transformation matrix
            transform = Quaternion(geom.quat).transformation_matrix
            transform[:3, 3] = geom.pos
            # apply transformation
            mesh.transform(transform)
            mesh_list.append(mesh)

    # if there are no meshes, return None
    if len(mesh_list) == 0:
        return None
    # else, merge all into one mesh
    else:
        return pv.merge(mesh_list)

def mj_get_contact_mesh(model, id):
    '''
    Given the MuJoCo model and body id, return the contact mesh of the body.
    '''
    # get body
    mj_body = model.body(id)
    # inspect geoms in the body
    mj_geom_adr = mj_body.geomadr[0]
    mj_geom_num = mj_body.geomnum[0]

    mesh = None
    for i in range(mj_geom_num):
        geom = model.geom(mj_geom_adr + i)
        if geom.dataid != -1 and geom.contype == 1:
            mesh = mj_mesh_to_polydata(model, geom.dataid)
            # convert quaternion to 4x4 transformation matrix
            transform = Quaternion(geom.quat).transformation_matrix
            transform[:3, 3] = geom.pos
            # apply transformation
            mesh.transform(transform)
            break

    return mesh

def mj_mesh_to_polydata(model, id):
    '''
    Given the MuJoCo model and the mesh id, return the PyVista PolyData mesh object.
    '''
    mj_mesh = model.mesh(id)
    # get points
    mj_point_range = slice(mj_mesh.vertadr[0], mj_mesh.vertadr[0] + mj_mesh.vertnum[0])
    mj_points = np.array(model.mesh_vert[mj_point_range])
    # get faces
    mj_face_range = slice(mj_mesh.faceadr[0], mj_mesh.faceadr[0] + mj_mesh.facenum[0])
    mj_faces = np.array(model.mesh_face[mj_face_range])

    '''
    PyVista PolyData takes in points array and faces array
    PolyData is a generalization of triangular meshes,
        faces array has each row in format [count, idx0, idx1, ...]
    Our faces array just has each row as [idx0, idx1, idx2]
    We need to convert it to [3, idx0, idx1, idx2]
    '''
    faces = np.hstack([
        np.full((mj_faces.shape[0], 1), 3), # n x 1 of 3's
        mj_faces # n x 3 of faces
    ])
    # create polydata object
    pv_mesh = pv.PolyData(mj_points, faces)

    return pv_mesh

def mj_get_body_transform(data, id: int):
    '''
    Given MuJoCo data and body id, return the transformation matrix of the body.
    '''
    # get position and rotation matrix
    pos = data.xpos[id]
    mat = data.xmat[id].reshape(3, 3)

    # create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = mat
    transform[:3, 3] = pos

    return transform

if __name__ == '__main__':
    # load panda arm model
    model = load_robot_description('panda_mj_description', variant='scene')
    data = mujoco.MjData(model)

    # print some model information
    print(f'Number of geom: {model.ngeom}')
    print(f'Number of body: {model.nbody}')
    print(f'Body Names: {[model.body(i).name for i in range(model.nbody)]}')
    print(f'Number of mesh: {model.nmesh}')
    print(f'Mesh Names: {[model.mesh(i).name for i in range(model.nmesh)]}')
    print(f'Time Step: {model.opt.timestep}')

    # get mesh names and visualize all the mesh in pyvista
    body_names = [model.body(i).name for i in range(model.nbody)]
    for name in body_names:
        mesh = mj_get_body_mesh(model, name)
        print('---------------')
        print(f'Visualize mesh {name}')
        print('---------------')
        if mesh != None:
            mesh.plot(style='wireframe')
