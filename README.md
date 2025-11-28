# Secure Multi-Party Computation

This project implements a Secure-Multi Party Computation (SMPC) for running K-means cluster on sensitive datasets while preserving the privacy of each participants. The protocol simulates a collaboration between two hospitals jointly analyzing lung cancer patient data without exposing their private datasets to each other.

# Project Overview
The project has the goal to perform K-means clustering across multiple parties without revealing a new data (Individual records remain hidden, and only necessary datapoints/shares are exchanged).


**Scenario:**
- Hospital A and Hospital B each hold private patient records.
- they want to cluster lung cancer patient data to extract confidential medical information.
- A Third-party, SMPC provides facilitates the ability to perform the computation.
- Dataset: UCI Lung cancer dataset - https://archive.ics.uci.edu/dataset/62/lung+cancer

# Protocol design
1. Share generation
- Each hospital splits its data points into randomized shares.
- Shares are exchange so no party see directly any data points/values.
2. Secure distance computation and assignment
- Distances to centroids are computed collaboratively using the shares.
- Each party performs only partial computations to preserve privacy.
3. Centroid share updates
- Each iteration updates cluster centroids using shares exchanged between hospitals.
- Random numbers cancel out, ensuring the final centroid is equivalent to plaintext K-means.
4. Convergence check
- Iterates until centroids stabilize or max iterations reached.

# Results
1. Time cost: ~7 seconds for 100 iterations
2. Communication cost:
- 36 KB sent and ~137 KB received per party.
- Communication grows with number of data points N, cluster K, and iterations I.
