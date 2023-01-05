#!/usr/bin/env python
# Matteo Lulli, Chi-Hang Lam 
# Applied Physics, Polytechnic University of Hong Kong, 2019
#Anupam modified it for put particles in hexagonal lattice
#XiaoChu add the attraction potential for one particle 

from random import shuffle
import random
import numpy as np
import sys

input_file = "data_polydisperse_hex2dmie.init"
output_diam = "diamaters_hex2dmie.init"
output_script = "generated_hex2dmie.lmp"
#fix_mark = "### Line for inserting the swap fixes (do not remove!)"
epsilon_lj = 1. # Couplings Default Value
gamma_rep = 12
gamma_att = 0
diameters_dump = True
Version = 1

def main():
    if len(sys.argv) != 9:
        print ("usage: python", sys.argv[0], " <zrange> <L> <a0> <width> <mean> <pair_cutoff> <epsilon_na> <types_number>")
        exit()

    # Assigning values from options
    zrange = float(sys.argv[1])
    L = int(sys.argv[2])
    a0 = float(sys.argv[3])
    width = float(sys.argv[4])
    mean = float(sys.argv[5])
    pair_cutoff = float(sys.argv[6])
    epsilon_na = float(sys.argv[7])
    types_number = int(sys.argv[8])

    # Setting Area and Volume
    A = L*L
    d_min = set_d_min(mean, width)
    atoms_number = int(1*A)
    if atoms_number % types_number != 0:
        print ("Error: the number of atoms (1A) must be a multiple of the number of types")
        exit()
    
    # Setting Diameters Values Distributed as Power-Law
    diameters = set_diameters(types_number, d_min, width)

    # If flag diameters_dump = True dump diameters for direct inspection
    if diameters_dump:
        with open(output_diam, "w") as diam_out:
            for type_index in range(types_number):
                diam_out.write("%d %022.15e %022.15e\n" %
                               (type_index + 1, diameters[type_index],
                                np.log(diameters[type_index])))

                               
    # Generating Data File
    generate_data_file(atoms_number, types_number, input_file, diameters,
                       zrange, L, a0, width, mean, epsilon_na, pair_cutoff)

    return

def cumulative_m1(x, d_min, width):
    r = d_min + x*width
    return r

def set_d_min(mean, width):
    d_min = mean - width/2.
    return d_min

def cong(b, X, c, m):
    return (b*X + c) % m

def set_diameters(types_number, d_min, width):
    X = 5
    diameters = list(range(types_number))
    newarr = np.zeros(types_number, float)
    for diameter_number in range(types_number):
        cumulative = float(diameter_number)/float(types_number)
        diameters[diameter_number] = cumulative_m1(cumulative, d_min, width)
    for i in range(types_number, 0, -1):
        index = 128 - i
        X = cong(1664525, X, 1013904223, 2**32)
        place = X % i
        newarr[index] = diameters[place]
        diameters = np.delete(diameters, place)
    return newarr
            
def generate_data_file(atoms_number, types_number, input_file, diameters,
                       zrange, L, a0, width, mean, epsilon_na, pair_cutoff):
    A = L*L
    # Generate Randomly Shuffled atoms -> type indices
    # Types always begin from 1

    # Random Shuffling
    #np.random.shuffle(atom_types_map)

    # Create Input Data File
    with open(input_file, "w") as lammps_input:
        # Inpute file Header
        lammps_input.write("L2d Polydisperse data file,  Parameters: <zrange>= %f <L>= %d <a0>= %f <width>= %f <mean>= %f <pair_cutoff> %f \
                            <epsilon_na>= %f <types_number>= %d \n" % (zrange, L, a0, width, mean, pair_cutoff, epsilon_na, types_number))

        lammps_input.write("\n")
        # Number of atoms and number of types
        lammps_input.write("%d atoms\n" % atoms_number)
        lammps_input.write("%d atom types\n" % types_number)
        lammps_input.write("\n")
        # Set Geometry bounds
        lammps_input.write("%022.15e %022.15e xlo xhi\n" % (0., a0*L))
        lammps_input.write("%022.15e %022.15e ylo yhi\n" % (0., a0*L))
        lammps_input.write("%022.15e %022.15e zlo zhi\n" % (-a0*zrange/2, a0*zrange/2))
        lammps_input.write("\n")
        # Masses Values
        #lammps_input.write("Masses\n\n")
        #for atom_type in range(types_number):
        #    lammps_input.write("%d 1.\n" % (atom_type + 1))

        # mie Potential Type Pairs Parameters (not atoms pairs!) 
        lammps_input.write("PairIJ Coeffs # mie/cut\n\n")
        for atom_type_i in range(types_number+1):
            for atom_type_j in range(atom_type_i, types_number+1):
                sigma_ij = 0.5*(diameters[atom_type_i] + diameters[atom_type_j])
                sigma_ij = sigma_ij*(1. - epsilon_na*np.abs(diameters[atom_type_i] -
                                                            diameters[atom_type_j]))
                cutoff_ij = sigma_ij*pair_cutoff
                lammps_input.write("%d %d %022.15e %022.15e %d %d %022.15e\n" %
                                   ((atom_type_i + 1), (atom_type_j + 1), epsilon_lj,
                                    sigma_ij, gamma_rep, gamma_att, cutoff_ij))
        lammps_input.write("\n")
        # Atoms Positions hexagonal
        lammps_input.write("Atoms # atomic\n\n")
        r_new = [[0]*3]*(1*A)
        atom_index, atom_type  = 1, 1
        for unit_cell_index in range(A):
            unit_cell_x = unit_cell_index % L
            unit_cell_y = unit_cell_index // L
#           print(unit_cell_x, unit_cell_y)

            rand1 = random.randint(1,50)/100.
            rand2 = random.randint(1,50)/100.
            
            if unit_cell_y % 2 != 0:
                r_new[unit_cell_index]   = [ a0*(unit_cell_x+rand1), a0*(unit_cell_y+rand2), 0]
            else:
                r_new[unit_cell_index]   = [ a0*(unit_cell_x+rand1)+0.5, a0*(unit_cell_y+rand2), 0]

        
        shuffle(r_new)

        for x in range(A*1):
            lammps_input.write("%d %d %022.15e %022.15e %022.15e 0 0 0 \n" %
                    (atom_index, atom_type, r_new[x][0], r_new[x][1], r_new[x][2] ))
            atom_index = atom_index + 1
            atom_type = atom_type + 1
            if atom_type == types_number + 1:
                atom_type = 1
             

print("output file: data_polydisperse_hex2dmie_init")

if __name__ == '__main__':
    main()
