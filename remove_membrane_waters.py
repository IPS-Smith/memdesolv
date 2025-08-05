import numpy as np
import MDAnalysis as mda
from scipy.interpolate import Rbf
from MDAnalysis.analysis.leaflet import LeafletFinder
from matplotlib import pyplot as plt
import argparse


# Argparse to include contra config file
parser = argparse.ArgumentParser(
            prog='condition tracker',
            description='',
            epilog='')

parser.add_argument('-f', '--stucture_file', type=str, help='gro file containing the structure of the solvated system with waters in the membrane')
parser.add_argument('-g', '--headgroup_selection', type=str, help='selection string for the headgroup atoms of the membrane (e.g. "resname DPPC and name PO4")')

args = parser.parse_args()


u = mda.Universe(args['stucture_file'])


#   Pick *one* representative bead/atom per lipid to build the graph.
headgroups = u.select_atoms(f"{args['headgroup_selection']}")

#   Build the connectivity graph and split it into connected components
lf = LeafletFinder(u, headgroups, cutoff=25.0, pbc=True)

#  Extract the two leaflets as AtomGroups.
leaflet_A, leaflet_B = lf.groups()         # returns a list of AtomGroups


# ── 4.  Sanity check & use them ────────────────────────────────────────
print(f"Lipids per leaflet: {leaflet_A.n_residues} / {leaflet_B.n_residues}")

def fit_smooth_surface(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # RBF interpolation for smooth surface fitting; 'thin_plate' allows wave-like deformations
    rbf = Rbf(x, y, z, function='thin_plate')
    return rbf

surface_A = fit_smooth_surface(leaflet_A.positions)
surface_B = fit_smooth_surface(leaflet_B.positions)

x_dims = [0, u.dimensions[0]]
y_dims = [0, u.dimensions[1]]

x_grid, y_grid = np.meshgrid(np.linspace(x_dims[0],x_dims[1],500), np.linspace(y_dims[0],y_dims[1],500))




if np.all(leaflet_A.positions[:, 2] > leaflet_B.positions[:, 2]):
    print("Leaflet A is above Leaflet B")
    upper_surface = surface_A
    lower_surface = surface_B
else:
    print("Leaflet B is above Leaflet A")
    upper_surface = surface_B
    lower_surface = surface_A


# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Surface plot
# ax.plot_surface(x_grid, y_grid, upper_surface(x_grid, y_grid), cmap='viridis', alpha=0.8)
# ax.plot_surface(x_grid, y_grid, lower_surface(x_grid, y_grid), cmap='copper', alpha=0.8)
# plt.show()

waters = u.select_atoms("resname W PW")

# 3.  PBC‑wrap x/y into the unit cell so they match the surface domain
box_x, box_y, _ = u.dimensions[:3]
coords = waters.positions.copy()
coords[:, 0] %= box_x
coords[:, 1] %= box_y

# 4.  Boolean mask: keep waters whose z is *outside* the bilayer
outside_mask = (
    (coords[:, 2] > upper_surface(coords[:, 0], coords[:, 1])) |  # above top leaflet
    (coords[:, 2] < lower_surface(coords[:, 0], coords[:, 1]))    # below bottom leaflet
)

waters_outside = waters[outside_mask]
print(f"Waters kept (outside bilayer): {waters_outside.n_atoms}")

waters_outside_full = waters_outside.residues.atoms

# 5.  Write them to PDB
with mda.Writer("waters_memdesolv.pdb", waters_outside_full.n_atoms) as W:
    W.write(waters_outside_full)