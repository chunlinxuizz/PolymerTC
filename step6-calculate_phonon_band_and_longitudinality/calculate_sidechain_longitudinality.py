import numpy as np
import glob
import re

natoms = 108
all_atoms = [ i for i in range(1,natoms+1) ]
backbone_atoms = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,43,44,45,46,105,106,107,108]
sidechain_atoms = [ i for i in all_atoms if i not in backbone_atoms ]


A = np.array([2.5272440003,     1.9016798529,   -18.7374369983])
B = np.array([0.6907920972,    -8.8914910907,     0.0000000000])
C = np.array([-5.5400380000,     0.0000000000,     0.0000000000])

volume = np.dot(A, np.cross(B, C))
a_star = 2 * np.pi * np.cross(B, C) / volume
b_star = 2 * np.pi * np.cross(C, A) / volume
c_star = 2 * np.pi * np.cross(A, B) / volume
G_star = np.array([a_star, b_star, c_star]).T


def calculate_longitudinality(q_vector, eigenvectors):
    q_vector_norm = np.linalg.norm(q_vector)
    total_longitudinality = 0.0
    num_atoms = len(eigenvectors)
    for ia,eigenvector in enumerate(eigenvectors):
        if ia in sidechain_atoms:
            dot_product = np.dot(q_vector, eigenvector)
            longitudinality = dot_product / (q_vector_norm * np.linalg.norm(eigenvector))
            total_longitudinality += longitudinality

    average_longitudinality = total_longitudinality / len(sidechain_atoms)
    return abs(average_longitudinality)**2

qpoint_files = sorted(glob.glob("qpoint_*.dat"), key=lambda x: int(re.findall(r'\d+', x)[0]))
total_files = len(qpoint_files)

results = []

for i, qpoint_file in enumerate(qpoint_files):
    print(f"Processing file {i + 1}/{total_files}: {qpoint_file}")
    
    with open(qpoint_file, 'r') as f:
        lines = f.readlines()
        q_vector = None
        distance = None
        eigenvectors_per_band = []
        frequencies = []
        current_eigenvectors = []
    
        for line in lines:
            if "q-position" in line:
                q_vector = np.array([float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)])
            elif "distance" in line:
                distance = float(re.findall(r"[-+]?\d*\.\d+", line)[0])
            elif "frequency" in line:
                frequency = float(re.findall(r"[-+]?\d*\.\d+", line)[0])
                frequencies.append(frequency)
                if current_eigenvectors:
                    eigenvectors_per_band.append(np.array(current_eigenvectors).reshape((natoms, 3)))
                    current_eigenvectors = []
            elif "eigenvector" in line:
                continue
            elif re.match(r'\s*-\s*\[\s*[-+]?\d*\.\d+,\s*[-+]?\d*\.\d+\s*\]', line):
                eigenvector_part = [float(x) for x in re.findall(r"[-+]?\d*\.\d+", line)]
                current_eigenvectors.append(eigenvector_part[0] + 1j * eigenvector_part[1])

        if current_eigenvectors:
            eigenvectors_per_band.append(np.array(current_eigenvectors).reshape((natoms, 3)))

        if q_vector is not None and eigenvectors_per_band:
            for j, eigenvectors in enumerate(eigenvectors_per_band):
                longitudinality = calculate_longitudinality(G_star@q_vector, eigenvectors)
                results.append((distance, frequencies[j], longitudinality))

output_file = "longitudinality_results_sidechain.dat"
num_bands = len(frequencies)

with open(output_file, "w") as f:
    for band_index in range(num_bands):
        for result in results[band_index::num_bands]:
            distance, frequency, longitudinality = result
            f.write(f"{distance} {frequency} {longitudinality}\n")
        f.write("\n\n")

print(f"Longitudinality calculation completed. Results saved to {output_file}.")

