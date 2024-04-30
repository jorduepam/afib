import pyvista as pv
import os
import numpy as np
from lapy import TriaMesh, Solver
import shutil
import json

# Function that defines a pyvista triangular mesh from vertices and faces matrix
def define_pyvista_mesh(vertices, faces):
    pv_faces = np.full((faces.shape[0], 4), 3, dtype=np.int_)
    pv_faces[:,1:] = faces
    return pv.PolyData(vertices, np.hstack(pv_faces)) 

def calculate_wall_movement(vertices, faces, t_values, dVla, NPVs):
    cycle_length = np.max(t_values)
    N_steps = len(t_values)  
    
    # Calculate timesteps
    T = np.linspace(0,cycle_length,N_steps+1)

    # Some initial checkings...
    surf_lapy_mesh = TriaMesh(vertices,faces)
    if surf_lapy_mesh.is_oriented() != True:
        raise NameError("WARNING, the mesh is not oriented.")
    if surf_lapy_mesh.is_manifold() != True:
        raise NameError("WARNING, the mesh is not manifold.")
    if surf_lapy_mesh.has_free_vertices() ==True:
        _,_=surf_lapy_mesh.rm_free_vertices_()
        vertices = surf_lapy_mesh.v
        faces = surf_lapy_mesh.t
        print('WARNING, removing detected free vertices.')

    # Define pyvista mesh
    pv_surf_mesh = define_pyvista_mesh(vertices, faces)
    surf_mesh_points = vertices
    surf_mesh_faces = faces
    
    # Extract boundary edges
    boundary_loops = surf_lapy_mesh.boundary_loops()
    if len(boundary_loops) != NPVs+1:
        raise NameError("ERROR, {0} boundary loops have been detected. Correct the mesh to have only {1} boundary loops.".format(len(boundary_loops), NPVs+1))
    # Assign biggest loop as MV
    mv_loop = [len(i) for i in boundary_loops].index(max([len(i) for i in boundary_loops]))
    mv_index = np.array(boundary_loops[mv_loop])
    pvs_index = np.concatenate([boundary_loops[i] for i in range(len(boundary_loops)) if i != mv_loop])

    # Calculate the mass center
    mass_center = surf_lapy_mesh.centroid()[0]

    # Some more checkings...
    genus_number = int((2 - surf_lapy_mesh.euler() - len(boundary_loops))/2)
    if genus_number != 0:
        raise NameError("ERROR, provided mesh is genus-{0}. Only genus-0 topologies are accepted, you should close existing holes.".format(genus_number))

    # Solve Laplace-Beltrami to get varphi
    fem = Solver(surf_lapy_mesh, lump=False) # lump: whether to lump the mass matrix (diagonal) 
    values = np.concatenate((np.zeros([len(pvs_index)]), np.ones([len(mv_index)])))
    idx = np.concatenate((pvs_index, mv_index))
    pv_surf_mesh["varphi"] = fem.poisson(h=0.0, dtup=(idx, values), ntup=())

    aux_vec = pv_surf_mesh["varphi"].reshape(len(pv_surf_mesh["varphi"]),1)
    # Calculate psi
    pv_surf_mesh["psi"] = 2*pv_surf_mesh["varphi"]*(1-pv_surf_mesh["varphi"])/np.max(np.concatenate((aux_vec,1-aux_vec),axis=1),axis=1)
    # Calculate x - x_G
    pv_surf_mesh["x_xG"] = surf_mesh_points-mass_center
    # Calculate chamber motion
    pv_surf_mesh["f_ALE"] = pv_surf_mesh["x_xG"]*pv_surf_mesh["psi"].reshape([len(pv_surf_mesh["psi"]),1])
    # Calculate areas and normals
    pv_surf_mesh["tria_normals"] = surf_lapy_mesh.tria_normals()
    pv_surf_mesh["tria_areas"] = surf_lapy_mesh.tria_areas()

    # Calculate global F_ALE 
    pv_surf_mesh["F_ALE"] = pv_surf_mesh["f_ALE"]  
    # Convert F_ALE to tria values
    pv_surf_mesh["tria_F_ALE"] = pv_surf_mesh.point_data_to_cell_data()["F_ALE"]
    
    # Calculate wall movement
    g_ALE = np.zeros([len(T), pv_surf_mesh["F_ALE"].shape[0], pv_surf_mesh["F_ALE"].shape[1]])
    mesh_volumes = np.zeros([len(T)])

    # Create VTK dir
    vtk_dir = ".\VTK"
    if os.path.isdir(vtk_dir):
        shutil.rmtree(vtk_dir)
    os.mkdir(vtk_dir)

    for Nstep in range(len(T)):
        # Calculate F_int y h_ALE
        F_int = 1/np.dot(np.diag(pv_surf_mesh["tria_F_ALE"]@(pv_surf_mesh["tria_normals"].transpose())),pv_surf_mesh["tria_areas"])
        try:
            h_ALE = F_int * dVla[Nstep]
        except:
            h_ALE = F_int * dVla[0]

        # Calculatete g_ALE
        g_ALE[Nstep] = pv_surf_mesh["F_ALE"]*h_ALE

        # Update points, normals and areas
        pv_surf_mesh.points = pv_surf_mesh.points + g_ALE[Nstep]*T[1]
        surf_lapy_mesh = TriaMesh(pv_surf_mesh.points,surf_mesh_faces)
        pv_surf_mesh["tria_normals"] = surf_lapy_mesh.tria_normals()
        pv_surf_mesh["tria_areas"] = surf_lapy_mesh.tria_areas()

        # Save surface meshes as .vtk
        pv_surf_mesh.save("{0}\surf_{1}.vtk".format(vtk_dir, Nstep))  

    dict_to_json = {"file-series-version" : "1.0", "files":[{"name":"surf_{0}.vtk".format(Nstep),"time":T[Nstep]} for Nstep in range(len(T))]}
    with open("{0}\surf.vtk.series".format(vtk_dir), 'w') as fp:
        json.dump(dict_to_json, fp, indent="\t")

    print("\n### Congratulations, process successfully completed!")
