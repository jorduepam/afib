import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pickle
import os
import scipy.io
import copy
from itertools import combinations
from scipy.signal import argrelextrema, savgol_filter
from matplotlib import tri as tri
from scipy import interpolate
import time
from pathlib import Path

import pyvista as pv
import potpourri3d as pp3d
from lapy import TriaMesh, Solver

pv.set_plot_theme("document")
pv.set_jupyter_backend('ipyvtklink')

# Function that saves object to file
def obj_to_file(obj, file_name):
    with open(file_name, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Funtion that loads object from file
def file_to_obj(file_name):
    if os.path.isfile(file_name) == True:
        with open(file_name, 'rb') as inp:
            return pickle.load(inp)
    else:
        print("Fichero {0} no encontrado".format(file_name))
        return dict()

# Function that return connectivity list of a segment
def find_connectivity(line_mesh, init_index, last_index=False):
    lines = line_mesh.lines.reshape(-1, 3)
    try:
        lines[0]
    except:
        lines = line_mesh.faces.reshape(-1, 3)
    conn = dict(lines[:, 1:])

    this_ind = init_index
    path_inds = [init_index]
    while True:
        next_ind = conn[this_ind]
        path_inds.append(next_ind)
        this_ind = next_ind
        if this_ind == init_index:
            break
    if last_index == True:
        return path_inds
    else:
        return path_inds[:-1]

# Function that returns part of a circular segment, containing the indicated indices
def divide_segment_from_index_array(line_mesh, point_idx_array, circular_edge = False):
    # Define new mesh
    poly = pv.PolyData()
    poly.points = line_mesh.points[point_idx_array]
    # Calculate connectivities
    connectivities = find_connectivity(line_mesh, point_idx_array[0], last_index=True)
    # Map new indices
    index_dict = {point_idx_array[i]:i for i in range(len(point_idx_array))}
    # Calculate new connectivities
    if circular_edge == True:
        cells = np.full((len(point_idx_array), 3), 2, dtype=np.int_)
        for i in range(len(point_idx_array)):
            cells[i, 1] = index_dict[connectivities[i]]
            cells[i, 2] = index_dict[connectivities[i+1]]
    else:
        cells = np.full((len(point_idx_array)-1, 3), 2, dtype=np.int_)
        for i in range(len(point_idx_array)-1):
            cells[i, 1] = index_dict[connectivities[i]]
            cells[i, 2] = index_dict[connectivities[i+1]]
    poly.lines = cells
    return poly

# Function that takes a line and reorder its connectivities
def reorder_line_according_connectivities(mesh_line, init_index=0, circular = False, return_conn = False):
    # Extract connectivities
    conn = find_connectivity(mesh_line, init_index, last_index=False)
    # Reorder vertices
    reordered_vertices = mesh_line.points[conn]
    # Define new connectivities
    reordered_faces = np.full((mesh_line.points.shape[0]-1, 3), 2, dtype=np.int_)
    reordered_faces[:,1] = np.arange(mesh_line.points.shape[0]-1)
    reordered_faces[:,2] = reordered_faces[:,1] + 1
    if circular == True:
        reordered_faces = np.vstack([reordered_faces, np.array([[2, mesh_line.points.shape[0]-1, 0]])])
    if return_conn == True:
        return pv.PolyData(reordered_vertices, np.hstack(reordered_faces)), conn
    else:
        return pv.PolyData(reordered_vertices, np.hstack(reordered_faces))

# Function that convert points array to a connected mesh lines
def convert_points_to_connected_line(points_array, circular = False):
    # Define new mesh
    poly_mesh = pv.PolyData()
    poly_mesh.points = points_array
    cells = np.full((len(points_array)-1, 3), 2, dtype=np.int_)
    cells[:,1] = np.arange(len(points_array)-1)
    cells[:,2] = cells[:,1] + 1
    if circular == True:
        cells = np.vstack([cells, np.array([[2, len(points_array)-1, 0]])])
    poly_mesh.lines = cells
    return poly_mesh

# Function that returns dist and idx of the N nearest points in a cloud of points 
def return_N_nearest_points_in_mesh(point_array, point_to_evaluate, N_points=1):
    tree = KDTree(point_array)
    pts = point_to_evaluate.reshape([1, point_to_evaluate.shape[0]])
    dist, idx = tree.query(pts, k=N_points)
    return dist, idx

# Function that divides circular edge in two segments
def divide_circular_edge_in_two_segments(mesh_line, init_point, end_point):
    init_point_idx = mesh_line.find_closest_point(init_point)
    end_point_idx = mesh_line.find_closest_point(end_point)

    # Recorre edge en un sentido
    connected_index = find_connectivity(mesh_line, init_point_idx, last_index=True)
    first_point_index_list = list()
    for index in connected_index:
        first_point_index_list += [index]
        if index == end_point_idx:
            break
    first_segment = divide_segment_from_index_array(mesh_line, first_point_index_list, circular_edge = False)

    # Recorre edge en sentido contrario
    connected_index = find_connectivity(mesh_line, end_point_idx, last_index=True)
    second_point_index_list = list()
    for index in connected_index:
        second_point_index_list += [index]
        if index == init_point_idx:
            break
    second_segment = divide_segment_from_index_array(mesh_line, second_point_index_list, circular_edge = False)
    
    return first_segment, second_segment, first_point_index_list, second_point_index_list

# Function that reverse line points if init point does not match
def reverse_connected_line_according_init_point(mesh_line, init_point, idx_list=[]):
    if np.linalg.norm(mesh_line.points[0]-init_point)>np.linalg.norm(mesh_line.points[-1]-init_point):
        mesh_line.points = np.flip(mesh_line.points, 0).copy() #  mesh_line.points[::-1]
        if idx_list != []:
            idx_list.reverse()
    return mesh_line, idx_list

# Function that, given integer keys and values, remap a input array
def remap_int_array(keys, values, input_array):
    mapping_ar = np.zeros(keys.max()+1,dtype=values.dtype) 
    mapping_ar[keys] = values
    return mapping_ar[input_array]

# Function that defines a pyvista triangular mesh from vertices and faces matrix
def define_pyvista_mesh(vertices, faces):
    pv_faces = np.full((faces.shape[0], 4), 3, dtype=np.int_)
    pv_faces[:,1:] = faces
    return pv.PolyData(vertices, np.hstack(pv_faces)) 
     
def perform_UAC(patient_name, show_figures = False, save_figures = True, return_results = True, alpha_remmaping=True, target_UAC=False):
       
    print("\n###############################")
    print("# Universal atrial coordinates")
    print("# Patient name: {0}".format(patient_name))
    print("###############################")    
    
    start_time = time.time()
    # File existence checkings
    if Path("{0}".format(patient_name)).is_dir() != True:
        raise NameError("ERROR, {0} folder not found.".format(patient_name))
    if Path("{0}\\case{0}_surfmesh.mat".format(patient_name)).exists() != True:
        raise NameError("ERROR, {0} not found.".format(Path("{0}\\case{0}_surfmesh.mat".format(patient_name))))
    if Path("{0}\\{0}_points.txt".format(patient_name)).exists() != True:
        raise NameError("ERROR, {0}\\{0}_points.txt not found.".format(patient_name))

    # Load landmark points from file
    points_dict = dict()
    point_labels = ["LAA_tip", "LAA_base", "FO", "rspv", "lspv", "lipv", "ripv", "fifthpv"]
    points_array = np.loadtxt(Path("{0}\\{0}_points.txt".format(patient_name)), comments='#', delimiter=",")
    for i in range(points_array.shape[0]):
        points_dict[point_labels[i]] = points_array[i] 
    PV_labels = list(points_dict.keys())[3:]

    ## Load triangular surfmesh .mat
    mat = scipy.io.loadmat(Path("{0}\\case{0}_surfmesh.mat".format(patient_name)))
    # Extract vertices
    vertices = mat["surfmesh"]["vertices"][0,0]
    # Extract faces
    faces = mat["surfmesh"]['faces'][0,0]
    if np.min(faces) == 1:
        faces = faces - 1

    # Some initial checkings...
    lapy_mesh = TriaMesh(vertices,faces)
    if lapy_mesh.is_oriented() != True:
        raise NameError("WARNING, the mesh is not oriented.")
    if lapy_mesh.is_manifold() != True:
        raise NameError("WARNING, the mesh is not manifold.")
    if lapy_mesh.has_free_vertices() ==True:
        _,_=lapy_mesh.rm_free_vertices_()
        vertices = lapy_mesh.v
        faces = lapy_mesh.t
        print('WARNING, removing detected free vertices.')

    # Construct pyvista mesh object
    la_mesh = define_pyvista_mesh(vertices, faces)

    # Extract boundary edges
    boundary_loops = lapy_mesh.boundary_loops()
    if len(boundary_loops) != len(PV_labels)+1:
        raise NameError("ERROR, {0} boundary loops have been detected. In this case it should be {1} plus one more for the MV. Correct the mesh to have only {2} boundary loops.".format(len(boundary_loops), len(PV_labels), len(PV_labels)+1))
    # Assign each PV boundary edge according to the nearest PV landmark point
    boundary_dict = dict()
    boundary_list = list()
    boundary_idx_dict = dict()
    # Loop over the PV
    for current_PV in PV_labels:
        # Loop over de edges calculating the minimum distance to the PV landmark
        dist_array = np.zeros([len(PV_labels)+1])
        for i in range(len(PV_labels)+1):
            dist, idx = return_N_nearest_points_in_mesh(vertices[boundary_loops[i]], points_dict[current_PV], N_points=1)
            dist_array[i] = dist[0]
        # Assign each edge according to the nearest PV landmark point
        boundary_list.append(np.argmin(dist_array))
        boundary_dict[current_PV] = convert_points_to_connected_line(vertices[np.array(boundary_loops[np.argmin(dist_array)])], circular = False)
        boundary_idx_dict[current_PV] = np.array(boundary_loops[np.argmin(dist_array)])
    if len(boundary_list) > len(set(boundary_list)):
        point_labels = [[points_dict[point_name] for point_name in points_dict], [point_name for point_name in points_dict]]
        plot_4_view_atrial_scalars([la_mesh], point_labels=point_labels, show_fig = True, save_fig = False)
        raise NameError("ERROR: Incorrect PV edges-point matching. Please select each landmark point separated from the rest of PV.")
    else:
        # Assign the remaining egde as the MV edge
        mv_idx = [item for item in [i for i in range(len(PV_labels)+1)] if item not in boundary_list][0]
        boundary_dict["mv"] = convert_points_to_connected_line(vertices[np.array(boundary_loops[mv_idx])], circular = True)
        boundary_idx_dict["mv"] = np.array(boundary_loops[mv_idx])

    # Some more checkings...
    genus_number = int((2 - lapy_mesh.euler() - len(boundary_loops))/2)
    if genus_number != 0:
        raise NameError("ERROR, provided mesh is genus-{0}. Only genus-0 topologies are accepted, you should close existing holes.".format(genus_number))
    print("# {0} boundary loops detected and assigned to {1}".format(len(boundary_loops), PV_labels+["mv"]))

    ### Calculate geodesic distance to LAA_tip
    print("# Calculating geodesic distances...")
    solver = pp3d.MeshHeatMethodDistanceSolver(vertices,faces)
    la_mesh["laa_distance"] = solver.compute_distance(la_mesh.find_closest_point(points_dict["LAA_tip"]))
    la_mesh["laa_distance"] = la_mesh["laa_distance"].clip(min=0)

    ### Calculate geodesic distance to each PV
    for current_PV in PV_labels:
        la_mesh["{0}_distance".format(current_PV)] = solver.compute_distance_multisource(list(boundary_idx_dict[current_PV]))
        la_mesh["{0}_distance".format(current_PV)] = la_mesh["{0}_distance".format(current_PV)].clip(min=0)

    ### Calculate geodesic distance to MV
    la_mesh["mv_distance"] = solver.compute_distance_multisource(list(boundary_idx_dict["mv"]))
    la_mesh["mv_distance"] = la_mesh["mv_distance"].clip(min=0)
    print("DONE")

    # Calculate regions cutt-off
    cutoff_dict = dict()
    cutoff_dict["laa"] = la_mesh["laa_distance"][la_mesh.find_closest_point(points_dict["LAA_base"])]
    for current_PV in PV_labels:
        cutoff_dict[current_PV] = la_mesh["{0}_distance".format(current_PV)][la_mesh.find_closest_point(points_dict[current_PV])]

    # Define regions 
    edges_dict = dict()
    edges_idx_dict = dict()
    regions_dict = dict()
    regions_idx_dict = dict()
    for region in ["laa"]+PV_labels:
        print("# Defining {0} region...".format(region))
        # Filter index below cutoff
        regions_idx_dict[region] = np.argwhere(la_mesh["{0}_distance".format(region)]<cutoff_dict[region]).flatten()
        # Filter faces with 3 valid vertices
        region_faces = faces[np.sum(np.isin(faces, regions_idx_dict[region]), axis=1)==3]
        # Calculate region index in new region mesh
        region_faces_remmaping = remap_int_array(regions_idx_dict[region], np.arange(len(regions_idx_dict[region])), region_faces)   
        # Define pymesh regions
        regions_dict[region] = define_pyvista_mesh(vertices[regions_idx_dict[region]], region_faces_remmaping)
        # Checking region genus number
        lapy_region = TriaMesh(vertices[regions_idx_dict[region]],region_faces_remmaping)
        if int((2 - lapy_region.euler() - 2)/2) != 0:
            plot_4_view_atrial_scalars([regions_dict[region]], show_fig = True, save_fig = False)
            raise NameError("ERROR, {0} region is genus-{1}. You should change landmark points to obtain a valid region.".format(region, genus_number))
        # Look for boundary loops in each region
        edges_loops = TriaMesh(vertices[regions_idx_dict[region]],region_faces_remmaping).boundary_loops()
        if region == "laa":
            if len(edges_loops) != 1:
                plot_4_view_atrial_scalars([regions_dict["laa"]], show_fig = True, save_fig = False)
                raise NameError("ERROR, {0} boundary edge loops detected for LAA region.".format(len(edges_loops)))
            # Calculate LAA edge index in main mesh
            edges_idx_dict[region] = remap_int_array(np.arange(len(regions_idx_dict[region])), regions_idx_dict[region], edges_loops[0])
            edges_dict[region] = convert_points_to_connected_line(vertices[edges_idx_dict[region]], circular = True)
        else:
            if len(edges_loops) != 2:
                plot_4_view_atrial_scalars([regions_dict[region]], show_fig = True, save_fig = False)
                raise NameError("ERROR, {0} boundary edge loops detected for {1} region.".format(len(edges_loops), region))
            for closed_loop in range(len(edges_loops)):
                # Calculate PV edge index in main mesh
                index_loop = remap_int_array(np.arange(len(regions_idx_dict[region])), regions_idx_dict[region], edges_loops[closed_loop])
                # Select the loop not coincident with boundary edge
                if np.any(np.isin(boundary_idx_dict[region], index_loop)) == False:
                    edges_idx_dict[region] = index_loop
                    edges_dict[region] = convert_points_to_connected_line(vertices[edges_idx_dict[region]], circular = True)
        if not region in edges_idx_dict.keys():
            raise NameError("ERROR, {0} was impossible to assign.".format(region))
    # Check for intersections between regions
    for i in combinations(("laa",)+tuple(PV_labels), 2):
        if np.any(np.isin(regions_idx_dict[i[0]], regions_idx_dict[i[1]])) == True:
            plot_4_view_atrial_scalars([regions_dict[i[0]],regions_dict[i[1]]], point_labels=[[points_dict[i[0]], points_dict[i[1]]],[i[0],i[1]]], show_fig = True, save_fig = False)
            raise NameError("ERROR, intersection found between {0} and {1} regions. Please change landmark points.".format(i[0], i[1]))
    if np.any(np.isin(boundary_idx_dict["mv"], regions_idx_dict["laa"])) == True:
        plot_4_view_atrial_scalars([boundary_dict["mv"],regions_dict["laa"]], show_fig = True, save_fig = False)
        raise NameError("ERROR, intersection found between LAA region and MV. Please change LAA landmark points.")

    ## Define region "only_la" by substracting regions from la_mesh
    regions_idx_dict["only_la"] = np.array([], dtype=int)
    for region in regions_idx_dict:
        if region != "only_la":
            regions_idx_dict["only_la"] = np.concatenate((regions_idx_dict["only_la"], regions_idx_dict[region]))
    edges_idx = np.array([], dtype=int)
    for region in edges_idx_dict:
        edges_idx = np.concatenate((edges_idx, edges_idx_dict[region]))
    # Remove edges idx from regions idx
    regions_idx_dict["only_la"] = regions_idx_dict["only_la"][np.invert(np.isin(regions_idx_dict["only_la"], edges_idx))]
    # Remove regions idx from mesh idx
    regions_idx_dict["only_la"] = np.arange(len(vertices))[np.invert(np.isin(np.arange(len(vertices)), regions_idx_dict["only_la"]))]
    # Calculate "only_la" faces and remap new faces
    region_faces = faces[np.all(np.isin(faces, regions_idx_dict["only_la"]), axis=1)]
    region_faces_remmaping = remap_int_array(regions_idx_dict["only_la"], np.arange(len(regions_idx_dict["only_la"])), region_faces)
    # Define pymesh region "only_la"
    regions_dict["only_la"] = define_pyvista_mesh(vertices[regions_idx_dict["only_la"]], region_faces_remmaping)
    print("DONE")


    
    
    ### Paths calculation
    paths_dict = dict()
    paths_idx_dict = dict()

    ### Septal path calculation: MV -> FO -> RSPV
    print("# Calculating septal path...")
    la_mesh["FO_distance"] = solver.compute_distance(la_mesh.find_closest_point(points_dict["FO"]))
    # Calculate MV point closest to the FO
    mv_boundary_idx = np.argmin(la_mesh["FO_distance"][boundary_idx_dict["mv"]])
    init_septal_idx = boundary_idx_dict["mv"][mv_boundary_idx]
    # Calculate RSPV point closest to the FO
    rspv_edges_idx = np.argmin(la_mesh["FO_distance"][edges_idx_dict["rspv"]])
    end_septal_idx = edges_idx_dict["rspv"][rspv_edges_idx]
    # Calculate septal path
    path1 = la_mesh.geodesic(init_septal_idx, la_mesh.find_closest_point(points_dict["FO"]))
    path2 = la_mesh.geodesic(la_mesh.find_closest_point(points_dict["FO"]), end_septal_idx)
    paths_dict["septal"] = path1 + path2
    paths_idx_dict["septal"] = paths_dict["septal"]["vtkOriginalPointIds"]
    # Check for overlapping nodes between septal path and MV
    paths_idx_dict["septal"] = paths_idx_dict["septal"][np.sum(np.isin(paths_idx_dict["septal"], boundary_idx_dict["mv"]))-1:]
    paths_dict["septal"] = convert_points_to_connected_line(vertices[paths_idx_dict["septal"]], circular=False)

    ### RSPV-LSPV path calculation: from RSPV landmark to LSPV landmark 
    print("# Calculating RSPV-LSPV path...")
    init_rspv_lspv_point = edges_dict["rspv"].points[edges_dict["rspv"].find_closest_point(points_dict["rspv"])]
    end_rspv_lspv_point = edges_dict["lspv"].points[edges_dict["lspv"].find_closest_point(points_dict["lspv"])]
    paths_dict["rspv_lspv"] = la_mesh.geodesic(la_mesh.find_closest_point(init_rspv_lspv_point), la_mesh.find_closest_point(end_rspv_lspv_point))
    paths_idx_dict["rspv_lspv"] = paths_dict["rspv_lspv"]["vtkOriginalPointIds"]
    paths_dict["rspv_lspv"] = convert_points_to_connected_line(paths_dict["rspv_lspv"].points, circular=False)

    ### Lateral path
    print("# Calculating lateral path...")
    mv_idx_to_plot = dict()
    for region in PV_labels:
        mv_idx_to_plot[region] = np.argmin(la_mesh["{0}_distance".format(region)][boundary_idx_dict["mv"]])
    # Reorder MV connectivities 
    boundary_dict["mv"], conn = reorder_line_according_connectivities(boundary_dict["mv"], init_index=mv_idx_to_plot["rspv"], circular = True, return_conn=True)
    boundary_idx_dict["mv"] = boundary_idx_dict["mv"][conn]
    for region in PV_labels:
        mv_idx_to_plot[region] = np.argmin(la_mesh["{0}_distance".format(region)][boundary_idx_dict["mv"]])
    if not mv_idx_to_plot["lipv"]>mv_idx_to_plot["lspv"]:
        boundary_dict["mv"].points = np.flip(boundary_dict["mv"].points, 0).copy()
        boundary_idx_dict["mv"] = boundary_idx_dict["mv"][::-1]
        boundary_idx_dict["mv"] = np.concatenate((np.array(boundary_idx_dict["mv"][-1]).reshape([1]),boundary_idx_dict["mv"][0:-1]))
        for region in PV_labels:
            mv_idx_to_plot[region] = np.argmin(la_mesh["{0}_distance".format(region)][boundary_idx_dict["mv"]])
        print("Reversed MV direction")

    # Calculate distances to MV from nearest point in LSPV
    mv_distances = la_mesh["lspv_distance"][boundary_idx_dict["mv"]]
    # Calculate the nearest and furthest point of the ostium from the MV
    nearest_MV_ostium_index = edges_idx_dict["laa"][np.argmin(la_mesh["mv_distance"][edges_idx_dict["laa"]])]
    furthest_MV_ostium_index = edges_idx_dict["laa"][np.argmax(la_mesh["mv_distance"][edges_idx_dict["laa"]])]
    # Calculate the furthest point of the ostium segments from the two previous ones
    la_mesh["MV_ostium"] = solver.compute_distance_multisource([nearest_MV_ostium_index,furthest_MV_ostium_index])
    first_segment, second_segment, first_point_index_list, second_point_index_list = divide_circular_edge_in_two_segments(edges_dict["laa"], la_mesh.points[nearest_MV_ostium_index], la_mesh.points[furthest_MV_ostium_index])
    first_index = edges_idx_dict["laa"][first_point_index_list][np.argmax(la_mesh["MV_ostium"][edges_idx_dict["laa"][first_point_index_list]])]
    second_index = edges_idx_dict["laa"][second_point_index_list][np.argmax(la_mesh["MV_ostium"][edges_idx_dict["laa"][second_point_index_list]])]
    # Select the MV index to use as boundary
    first_MV_idx = np.argmin(solver.compute_distance(first_index)[boundary_idx_dict["mv"]])
    second_MV_idx = np.argmin(solver.compute_distance(second_index)[boundary_idx_dict["mv"]])
    if first_MV_idx > second_MV_idx:
        laa_posterior_mv_index = first_MV_idx
    else:
        laa_posterior_mv_index = second_MV_idx
    mv_idx = len(mv_distances[:laa_posterior_mv_index])+np.argmin(mv_distances[laa_posterior_mv_index:int(len(mv_distances)*3/4)])

    # Select the minimum between the local minimums
    minimum_indices = argrelextrema(mv_distances[laa_posterior_mv_index:int(len(mv_distances)*3/4)], np.less)[0]
    mv_idx = len(mv_distances[:laa_posterior_mv_index])+minimum_indices[np.argmin(mv_distances[laa_posterior_mv_index:int(len(mv_distances)*3/4)][minimum_indices])]


    # Calculate the nearest LSPV point
    lspv_edge_index = np.argmin(solver.compute_distance(boundary_idx_dict["mv"][mv_idx])[edges_idx_dict["lspv"]])
    # Calculate lateral path
    paths_dict["lateral"] = la_mesh.geodesic(edges_idx_dict["lspv"][lspv_edge_index], boundary_idx_dict["mv"][mv_idx])
    paths_idx_dict["lateral"] = paths_dict["lateral"]["vtkOriginalPointIds"]
    # Check for additional overlapping nodes between lateral path and MV
    lateral_path_MV_nodes = np.sum(np.isin(paths_idx_dict["lateral"], boundary_idx_dict["mv"]))
    if lateral_path_MV_nodes > 1:
        paths_idx_dict["lateral"] = paths_idx_dict["lateral"][:-(lateral_path_MV_nodes-1)]
    paths_dict["lateral"] = convert_points_to_connected_line(vertices[paths_idx_dict["lateral"]], circular=False)

    ### Divide RSPV and LSPV edges in posterior and anterior segments
    print("# Dividing RSPV and LSPV junctions in posterior and anterior segments...")
    # We will assign anterior segment to the one which is nearest to the LAA
    pv_segments_dict = dict()
    pv_segments_idx_dict = dict()
    LAA_edge_center = np.mean(edges_dict["laa"].points, axis=0)
    # Divide RSPV in two segments, anterior and posterior
    first_segment, second_segment, first_idx_list, second_idx_list = divide_circular_edge_in_two_segments(edges_dict["rspv"], paths_dict["septal"].points[-1], paths_dict["rspv_lspv"].points[0])
    if np.linalg.norm(LAA_edge_center - np.mean(first_segment.points, axis=0)) < np.linalg.norm(LAA_edge_center - np.mean(second_segment.points, axis=0)):
        pv_segments_dict["rspv_anterior"] = first_segment
        pv_segments_dict["rspv_posterior"] = second_segment
        pv_segments_idx_dict["rspv_anterior"] = first_idx_list
        pv_segments_idx_dict["rspv_posterior"] = second_idx_list
    else:
        pv_segments_dict["rspv_anterior"] = second_segment
        pv_segments_dict["rspv_posterior"] = first_segment
        pv_segments_idx_dict["rspv_anterior"] = second_idx_list
        pv_segments_idx_dict["rspv_posterior"] = first_idx_list
    # Divide LSPV in two segments, anterior and posterior
    first_segment, second_segment, first_idx_list, second_idx_list = divide_circular_edge_in_two_segments(edges_dict["lspv"], paths_dict["rspv_lspv"].points[-1], paths_dict["lateral"].points[0])
    if np.linalg.norm(LAA_edge_center - np.mean(first_segment.points, axis=0)) < np.linalg.norm(LAA_edge_center - np.mean(second_segment.points, axis=0)):
        pv_segments_dict["lspv_anterior"] = first_segment
        pv_segments_dict["lspv_posterior"] = second_segment
        pv_segments_idx_dict["lspv_anterior"] = first_idx_list
        pv_segments_idx_dict["lspv_posterior"] = second_idx_list
    else:
        pv_segments_dict["lspv_anterior"] = second_segment
        pv_segments_dict["lspv_posterior"] = first_segment
        pv_segments_idx_dict["lspv_anterior"] = second_idx_list
        pv_segments_idx_dict["lspv_posterior"] = first_idx_list
    # Reorder segments in case they were not in order 
    pv_segments_dict["rspv_posterior"], pv_segments_idx_dict["rspv_posterior"] = reverse_connected_line_according_init_point(pv_segments_dict["rspv_posterior"], paths_dict["septal"].points[-1], idx_list=pv_segments_idx_dict["rspv_posterior"])
    pv_segments_dict["lspv_posterior"], pv_segments_idx_dict["lspv_posterior"] = reverse_connected_line_according_init_point(pv_segments_dict["lspv_posterior"], paths_dict["rspv_lspv"].points[-1], idx_list=pv_segments_idx_dict["lspv_posterior"])
    pv_segments_dict["rspv_anterior"], pv_segments_idx_dict["rspv_anterior"] = reverse_connected_line_according_init_point(pv_segments_dict["rspv_anterior"], paths_dict["septal"].points[-1], idx_list=pv_segments_idx_dict["rspv_anterior"])
    pv_segments_dict["lspv_anterior"], pv_segments_idx_dict["lspv_anterior"] = reverse_connected_line_according_init_point(pv_segments_dict["lspv_anterior"], paths_dict["rspv_lspv"].points[-1], idx_list=pv_segments_idx_dict["lspv_anterior"])
    # Convert from edge index to mesh index
    pv_segments_idx_dict["rspv_posterior"] = edges_idx_dict["rspv"][pv_segments_idx_dict["rspv_posterior"]]
    pv_segments_idx_dict["lspv_posterior"] = edges_idx_dict["lspv"][pv_segments_idx_dict["lspv_posterior"]]
    pv_segments_idx_dict["rspv_anterior"] = edges_idx_dict["rspv"][pv_segments_idx_dict["rspv_anterior"]]
    pv_segments_idx_dict["lspv_anterior"] = edges_idx_dict["lspv"][pv_segments_idx_dict["lspv_anterior"]]
    # Define roof and base paths and indices
    paths_dict["roof"] = pv_segments_dict["rspv_posterior"]+paths_dict["rspv_lspv"] + pv_segments_dict["lspv_posterior"]
    paths_idx_dict["roof"] = np.concatenate((pv_segments_idx_dict["rspv_posterior"], paths_idx_dict["rspv_lspv"][1:]), axis=0)
    paths_idx_dict["roof"] = np.concatenate((paths_idx_dict["roof"], pv_segments_idx_dict["lspv_posterior"][1:]), axis=0)
    paths_dict["base"] = boundary_dict["mv"]
    paths_idx_dict["base"] = boundary_idx_dict["mv"]

    ## Final checkings
    for path in paths_idx_dict:
        index_list = paths_idx_dict[path]
        # Check connectivities for each path on la_mesh
        for vertex_pair in [index_list[i:i+2] for i in range(len(index_list)-1)]:
            if np.max(np.sum(np.isin(faces, vertex_pair, assume_unique=False), axis=1)) < 2:
                print("WARNING, vertex {0} of {1} path are not connected in la_mesh... that means problems.".format(vertex_pair, path))
        # Check if index for each path are unique
        if len(np.unique(index_list)) != len(index_list):
            print("WARNING, vertex index of {0} path are not unique... that means problems.".format(path))
    # Check if final vertex of septal path matches init vertex of roof path
    if paths_idx_dict["septal"][-1] != paths_idx_dict["roof"][0]:
        print("WARNING, final vertex of septal path does not match init vertex of roof path... that means problems.".format(path))
    # Check if final vertex of roof path matches init vertex of lateral path
    if paths_idx_dict["lateral"][0] != paths_idx_dict["roof"][-1]:
        print("WARNING, final vertex of roof path does not match init vertex of lateral path... that means problems.".format(path))
    print("DONE")

    ## Mesh division in anterior and posterior meshes
    print("# Dividing mesh in anterior and posterior parts...")
    # Mark the faces in the anterior part as 1 and the faces in the posterior part as 0
    la_mesh.cell_data['anterior'] = np.zeros([la_mesh.n_cells])
    # Mark RSPV and LSPV cells as anterior
    la_mesh.cell_data["anterior"][np.all(np.isin(faces, regions_idx_dict["rspv"]), axis=1)] = 1
    la_mesh.cell_data["anterior"][np.all(np.isin(faces, regions_idx_dict["lspv"]), axis=1)] = 1
    # Look for a cell in the LAA tip
    current_face = np.argmax(np.sum(np.isin(faces, la_mesh.find_closest_point(points_dict["LAA_tip"]), assume_unique=False), axis=1))
    la_mesh.cell_data['anterior'][current_face]=1
    borders_idx = np.concatenate((paths_idx_dict["septal"][:-1], paths_idx_dict["roof"][:-1], paths_idx_dict["lateral"]))
    faces_to_check = [current_face]
    # Construct the borders adjacency matrix
    borders_adjacency_matrix = np.concatenate((borders_idx[:-1].reshape([len(borders_idx)-1,1]),borders_idx[1:].reshape([len(borders_idx)-1,1])), axis=1)

    # Look for all border faces
    all_border_cells = np.argwhere(np.sum(np.isin(faces, borders_idx), axis=1))    

    # Function that return every point adjacent cell
    def find_adjacent_cells(cells, index):
        """Given the cells array, find other cells that 
        share vertices with the cell at ``index``"""
        cell = cells[index]
        adjacent_cells = np.argwhere(np.any((np.isin(cells, cell)), axis=1)).ravel()
        return adjacent_cells

    # Function designed to paint anterior cells with a scalar = 1
    def keep_moving(faces_to_check, anterior_border_faces):
        adjacent_cells = find_adjacent_cells(faces, faces_to_check)
        # Filter adjacent cells which are already classified
        adjacent_cells = adjacent_cells[np.isin(la_mesh.cell_data['anterior'][adjacent_cells],0.)]
        la_mesh.cell_data['anterior'][adjacent_cells]=1

        # Save in list anterior border faces for last layer
        anterior_border_faces += list(adjacent_cells[np.isin(adjacent_cells,all_border_cells)])

        # Exclude from next loop border cells
        adjacent_cells = adjacent_cells[np.invert(np.any(np.isin(faces[adjacent_cells], borders_idx), axis=1))]
        return list(adjacent_cells),anterior_border_faces

    anterior_border_faces = list()
    # Loop that paints anterior cells with a scalar = 1
    while faces_to_check != []:
        faces_to_check, anterior_border_faces = keep_moving(faces_to_check, anterior_border_faces)

    # Function that classificate as anterior adjacent cell by edges
    def extend_anterior_zone_by_edges(index):
        # Look for adjacent faces
        adjacent_faces = np.where(np.sum(np.isin(faces, faces[index]), axis=1)==2)[0]
        # Identify the connecting edges
        adjacent_edges = faces[adjacent_faces][np.isin(faces, faces[index])[adjacent_faces]].reshape(len(adjacent_faces),2)
        for i in range(len(adjacent_faces)):
            # Filter the edges which are present in borders
            if not np.any(np.all(np.isin(borders_adjacency_matrix, adjacent_edges[i]), axis=1)) and la_mesh.cell_data["anterior"][adjacent_faces[i]]==0:
                la_mesh.cell_data["anterior"][adjacent_faces[i]] = 1
                return adjacent_faces[i]

    # Second pass close to the border to ensure that all faces are classified as anterior
    for index in anterior_border_faces:
        new_index = extend_anterior_zone_by_edges(index)
        if new_index != None:
            extend_anterior_zone_by_edges(new_index)

    ## Define anterior mesh    
    # Look for anterior faces
    anterior_faces = faces[la_mesh.cell_data['anterior']==1]
    # Extract anterior vertices index
    anterior_idx_vertices = np.unique(anterior_faces.flatten())
    # Remap anterior faces to match anterior vertices
    anterior_local_faces = remap_int_array(anterior_idx_vertices, np.arange(len(anterior_idx_vertices)), anterior_faces)
    anterior_mesh = define_pyvista_mesh(vertices[anterior_idx_vertices], anterior_local_faces)

    ## Define posterior mesh
    # Look for posterior faces
    posterior_faces = faces[la_mesh.cell_data['anterior']==0]
    # Extract posterior vertices index
    posterior_idx_vertices = np.unique(posterior_faces.flatten())
    # Remap posterior faces to match posterior vertices
    posterior_local_faces = remap_int_array(posterior_idx_vertices, np.arange(len(posterior_idx_vertices)), posterior_faces)
    posterior_mesh = define_pyvista_mesh(vertices[posterior_idx_vertices], posterior_local_faces)

    # Some checkings 
    if np.all(np.isin(borders_idx, posterior_idx_vertices)) != True:
        print("WARNING, not all border indices are in posterior mesh... that means problems.")
    if np.all(np.isin(borders_idx, anterior_idx_vertices)) != True:
        print("WARNING, not all border indices are in anterior mesh... that means problems.")
    if anterior_mesh.points.shape[0] + posterior_mesh.points.shape[0] - borders_idx.shape[0] != la_mesh.points.shape[0]:
        print("WARNING, point count after mesh division does not match... that means problems.")
    print("DONE")

    
    
    # Coordinate beta (posterior-anterior) - Dirichlet conditions: 0 at the base path and 1 at the roof path
    print("# Calculating beta coordinate (posterior-anterior)...")
    fem = Solver(lapy_mesh, lump=False) # lump: whether to lump the mass matrix (diagonal) 
    values = np.concatenate((np.zeros([len(paths_idx_dict["base"])]), np.ones([len(paths_idx_dict["roof"])])))
    idx = np.concatenate((paths_idx_dict["base"], paths_idx_dict["roof"]))
    if target_UAC != False:
        values = np.concatenate((values, target_UAC["transformed_values"][:,1]))
        idx = np.concatenate((idx, target_UAC["transformed_idx"].flatten()))
    la_mesh["beta_coordinate"] = fem.poisson(h=0.0, dtup=(idx, values), ntup=()) # uac_results.keys()

    # Duplicate nodes in septal and lateral paths for anterior faces
    print("# Duplicating septal and lateral nodes...")
    new_index_to_add = np.concatenate((paths_idx_dict["septal"][:-1],paths_idx_dict["lateral"][1:]))
    new_points_numbering = np.arange(0, np.max(faces)+1+len(new_index_to_add))
    new_point_mapping = np.copy(new_points_numbering)
    new_points_idx = new_point_mapping[np.isin(new_points_numbering, new_index_to_add)]

    # Duplicate nodes in septal and lateral paths for anterior faces
    new_points_xyz = la_mesh.points[new_points_idx]
    new_vertices = np.concatenate((vertices, new_points_xyz))

    # Remap anterior faces to adjust to new anterior nodes
    new_point_mapping[np.isin(new_points_numbering, new_points_idx)] = np.arange(np.max(faces)+1, np.max(faces)+1+len(new_points_idx))
    new_faces = np.copy(faces)
    new_faces[la_mesh.cell_data['anterior']==1] = remap_int_array(new_points_numbering, new_point_mapping, faces[la_mesh.cell_data['anterior']==1])

    # Define new mesh
    new_la_mesh = define_pyvista_mesh(new_vertices, new_faces)
    new_la_mesh.cell_data["anterior"] = la_mesh.cell_data["anterior"]
    new_lapy_mesh = TriaMesh(new_vertices,new_faces)

    # Calculate beta values for duplicated nodes
    duplicated_beta_values = la_mesh["beta_coordinate"][new_points_idx]
    new_la_mesh.point_data["beta_coordinate"] = np.concatenate((la_mesh["beta_coordinate"], duplicated_beta_values))

    # Posterior-anterior beta coordinate inversion
    new_posterior_idx_vertices = np.unique(new_faces[new_la_mesh.cell_data['anterior']==0].flatten())
    new_la_mesh["beta_coordinate"][new_posterior_idx_vertices] = 0.5*new_la_mesh["beta_coordinate"][new_posterior_idx_vertices]
    new_anterior_idx_vertices = np.unique(new_faces[new_la_mesh.cell_data['anterior']==1].flatten())
    new_la_mesh["beta_coordinate"][new_anterior_idx_vertices] = 1-0.5*new_la_mesh["beta_coordinate"][new_anterior_idx_vertices]
    new_la_mesh["beta_coordinate"][paths_idx_dict["roof"]] = 0.5

    # Define new septal and lateral index
    new_septal_index = np.concatenate((paths_idx_dict["septal"], remap_int_array(new_points_numbering, new_point_mapping, paths_idx_dict["septal"][:-1])))
    new_lateral_index = np.concatenate((paths_idx_dict["lateral"], remap_int_array(new_points_numbering, new_point_mapping, paths_idx_dict["lateral"][1:])))

    # Coordinate alpha (lateral-septal) - Dirichlet conditions: 0 at septal path and 1 at lateral path
    print("# Calculating alpha coordinate (lateral-septal)...")
    fem = Solver(new_lapy_mesh, lump=False) # lump: whether to lump the mass matrix (diagonal) 
    values = np.concatenate((np.zeros([len(new_septal_index)]), np.ones([len(new_lateral_index)])))
    idx = np.concatenate((new_septal_index, new_lateral_index))
    if target_UAC != False:
        values = np.concatenate((values, target_UAC["transformed_values"][:,0]))
        idx = np.concatenate((idx, target_UAC["transformed_idx"].flatten()))
    new_la_mesh["alpha_coordinate_before_rescaling"] = fem.poisson(h=0.0, dtup=(idx, values), ntup=()) # , use_cholmod=True

    if alpha_remmaping == False:
        new_la_mesh["alpha_coordinate"] = new_la_mesh["alpha_coordinate_before_rescaling"]
    else:
        # Calculate iso-surfaces
        beta_contours = new_la_mesh.contour(isosurfaces=np.linspace(0,1,101), scalars="beta_coordinate", method='contour', progress_bar=False) 

        ## Calculate beta = 0.2 isoline
        beta_posterior_isoline = new_la_mesh.contour(isosurfaces=np.array([0.2]), scalars="beta_coordinate", method='contour', progress_bar=False) 
        # Sort isoline connectivities
        sorted_idx = np.argsort(beta_posterior_isoline["alpha_coordinate_before_rescaling"])
        alpha_coordinate = copy.copy(beta_posterior_isoline["alpha_coordinate_before_rescaling"][sorted_idx])
        beta_posterior_isoline = convert_points_to_connected_line(beta_posterior_isoline.points[sorted_idx], circular = False)
        beta_posterior_isoline["alpha_coordinate_before_rescaling"] = alpha_coordinate
        normalised_geodesic_distance = np.linspace(0,1,len(beta_posterior_isoline["alpha_coordinate_before_rescaling"]))

        # Smooth alpha by applying Savitzky-Golay filter
        if (int(len(alpha_coordinate)*2/3)) % 2 == 0:
            filtered_alpha_coordinate = savgol_filter(alpha_coordinate, int(len(alpha_coordinate)*2/3)+1, 4) 
        else:
            filtered_alpha_coordinate = savgol_filter(alpha_coordinate, int(len(alpha_coordinate)*2/3), 4) 
        # Rescale filtered alpha ends
        filtered_alpha_coordinate = (filtered_alpha_coordinate-filtered_alpha_coordinate[0])
        filtered_alpha_coordinate = filtered_alpha_coordinate/filtered_alpha_coordinate[-1]
        filtered_alpha_coordinate[-1]=1 

        ## Remapping values of alpha
        print("# Alpha coordinate rescaling...")
        mapping1 = interpolate.interp1d(filtered_alpha_coordinate, normalised_geodesic_distance)
        mapping2 = interpolate.interp1d(normalised_geodesic_distance, filtered_alpha_coordinate)
        new_la_mesh["alpha_coordinate"] = mapping1(new_la_mesh["alpha_coordinate_before_rescaling"])

    xyz_vertices = new_la_mesh.points
    ab_vertices = np.array([new_la_mesh["alpha_coordinate"], new_la_mesh["beta_coordinate"], np.zeros([len(new_la_mesh.points)])]).transpose()
    face_conn = new_lapy_mesh.t
    lapy_final_mesh = TriaMesh(xyz_vertices,face_conn)
    print("DONE")
    
    
    

    if target_UAC != False:
        results_folder = Path("{0}\\{0}_UAC_results_target_{1}".format(patient_name, target_UAC["target_patient_name"]))
    else:
        results_folder = Path("{0}\\{0}_UAC_results".format(patient_name))
    if results_folder.is_dir() != True:
        results_folder.mkdir()

    # Store UAC results in .mat y en .pkl
    print("# Saving results...")
    uac_results = {"xyz_vertices":xyz_vertices, "ab_vertices":ab_vertices, "faces":face_conn, "old_xyz_vertices":vertices, "old_faces":faces, "boundary_idx_dict":boundary_idx_dict, "edges_idx_dict":edges_idx_dict, "paths_idx_dict":paths_idx_dict, "anterior_faces":la_mesh["anterior"]}
    if target_UAC != False:
        obj_to_file(uac_results, Path("{0}\\{1}_UAC_results_target_{2}.pkl".format(results_folder, patient_name, target_UAC["target_patient_name"])))
    else:
        obj_to_file(uac_results, Path("{0}\\{1}_UAC_results.pkl".format(results_folder, patient_name)))

    print("# DONE: UAC for patient {0} completed in {1:.1f} seconds.\n".format(patient_name, time.time() - start_time))

    
    if return_results == True:
        return uac_results
    else:
        return

if __name__ == '__main__':
    find_connectivity()