import numpy as np
import pyvista as pv

from utility.mujoco_mesh_extract import mj_get_body_mesh, mj_get_body_transform

class PVInterface:
    def __init__(self, model, data):
        # record mj model and data
        self.model = model
        self.data = data
        # get copy of mesh for pyvista visualization
        self.body_meshes = {}
        self.body_mesh_points = {}
        for i in range(model.nbody):
            self.body_meshes[i] = mj_get_body_mesh(model, i)
            if self.body_meshes[i] is not None:
                self.body_mesh_points[i] = np.array(self.body_meshes[i].points)
        # initialize pyvista
        pv.set_plot_theme('document')
        self.pl = pv.Plotter()
        self.pl.add_axes()
        self.pl.show(interactive_update=True)
        # add meshes to pyvista
        self.body_mesh_actors = {}
        for i in range(model.nbody):
            if self.body_meshes[i] is not None:
                actor = self.pl.add_mesh(self.body_meshes[i],
                                         color='lightblue',
                                         show_edges=True,
                                         name=f'body_{i}')
                self.body_mesh_actors[i] = actor

    def track_body(self, body_name):
        # define body name & id
        self.body_name = body_name
        self.body_id = self.model.body(body_name).id
        # change mesh color to red
        self.body_mesh_actors[self.body_id].prop.color = 'red'

        # add visualization arrow
        arrow = pv.Arrow(start=(0, 0, 0), direction=(0, 0, 1), scale=1)
        self.arrow_actor = self.pl.add_mesh(arrow, color='red', name='arrow')

    def update_vector(self, vector, scale=0.1):
        position = mj_get_body_transform(self.data, self.body_id)[:3, 3]
        new_arrow = pv.Arrow(start=position, direction=vector, scale=scale * np.linalg.norm(vector, ord=2))
        self.arrow_actor.mapper.SetInputData(new_arrow)
        self.arrow_actor.mapper.update()

    def update_meshes(self):
        for i in range(self.model.nbody):
            if self.body_meshes[i] is not None:
                # update mesh points
                self.body_meshes[i].points = self.body_mesh_points[i]
                # update mesh transform
                self.body_meshes[i].transform(mj_get_body_transform(self.data, i))

    def pv_render(self):
        self.update_meshes()
        self.pl.update()
