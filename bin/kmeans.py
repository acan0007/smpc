import numpy as np
import sys
import socket
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

DIM = 57  # Dimensionality of the dataset
MAX_ITER = 100  # Maximum iterations
PREDETERMINED_K = 3 # Predetermine K value for k-means clustering

def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            if '?' in values or len(values) != DIM:
                print(f"Skipping line due to issues in data.")
                continue
            data.append(list(map(int, values)))
    return np.array(data)

def determine_k(data):
    return PREDETERMINED_K

def initialize_centroids(data, k):
    # Randomly initialize centroids
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

def calculate_distance(data, centroids):
    distances = np.zeros((len(data), len(centroids)))
    for i, d in enumerate(data):
        for j, c in enumerate(centroids):
            distances[i][j] = np.sum((d - c) ** 2)
    return distances

def assign_clusters(distances):
    return np.argmin(distances, axis=1)

def update_centroids(data, clusters):
    centroids = np.zeros((len(np.unique(clusters)), DIM))
    counts = np.zeros(len(np.unique(clusters)))
    for i, d in enumerate(data):
        idx = clusters[i]
        centroids[idx] += d
        counts[idx] += 1
    for j in range(len(centroids)):
        if counts[j] > 0:
            centroids[j] /= counts[j]
    return centroids

def compute_inertia(data, centroids, clusters):
    inertia = 0
    for i, d in enumerate(data):
        centroid = centroids[clusters[i]]
        inertia += np.sum((d - centroid) ** 2)
    #print(f"Computed Final Inertia: {inertia}")  # Debug print
    return inertia

def compute_max_inertia(data):
    # Calculate the overall mean of the dataset
    overall_mean = np.mean(data, axis=0)
    # Calculate the total sum of squared distances from each data point to the overall mean
    max_inertia = np.sum(np.sum((data - overall_mean) ** 2, axis=1))
    #print(f"Corrected Max Inertia: {max_inertia}")  # For debug print
    return max_inertia

def compute_accuracy(inertia, max_inertia):
    #print(f"Inertia: {inertia}, Max Inertia: {max_inertia}")  #For debug print
    if max_inertia == 0:
        return 0  # Prevent division of final acc by zero
    return max(0, min(1.0, 1.0 - (inertia / max_inertia)))  # Ensure accuracy is within range 0 - 1

def output_cluster_assignments(data, clusters, filename="data_clusters.csv"): # Print and save clustered data as csv file
    with open(filename, 'w') as f:
        for point, cluster in zip(data, clusters):
            f.write(f"{','.join(map(str, point))},{cluster}\n")
    print(f"Cluster assignments written to {filename}")

def plot_clusters(data, centroids, clusters):
    # Plot clustered data assignment using matplotlib
    plt.figure(figsize=(10, 6))
    for i in range(len(np.unique(clusters))):
        cluster_data = data[clusters == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', label='Centroids')
    plt.legend()
    plt.title('Clustered Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig("clusters.png")

def main(argv):
    if len(argv) != 4:
        print("Usage: python script.py <party_id> <port> <data_file>")
        return

    party_id = int(argv[1])
    port = int(argv[2])
    data_file = argv[3]

    # Start to timing the protocol
    protocol_start_time = time.time()

    party_id = int(argv[1])
    port = int(argv[2])
    data_file = argv[3]
    total_sent_bytes = 0
    total_received_bytes = 0

    # Function to print the party-X's protocol parameter input
    print("Running with settings:")
    print(f"Party ID: {party_id}")
    print(f"Port: {port}")
    print(f"Data file: {data_file}")

    # Function read the csv dataset/private input data points from each party
    data = read_data(data_file)
    if len(data) == 0:
        print("No valid data points loaded; please check your data file.")
        return

    # Party 1 - Hospital A side
    if party_id == 1:
        # Wait for party 2 to connect
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('localhost', port))
        s.listen(1)
        print("Waiting for party 2 to connect...")
        conn, _ = s.accept()
        print("Connected to party 2.")

        # Function to print out the predetermined K-value
        k = determine_k(data)
        print(f"Optimal K determined: {k}")

        centroids = initialize_centroids(data, k)

        # For loop for the iteration
        iteration_times = []
        for _ in range(MAX_ITER):
            start_time = time.time()
            distances = calculate_distance(data, centroids)
            clusters = assign_clusters(distances)

            # Serialize and send clusters to party 2
            serialized_clusters = pickle.dumps(clusters)
            conn.send(serialized_clusters)
            total_sent_bytes += len(serialized_clusters)

            # Receive and deserialize updated centroids from party 2
            received_centroids = conn.recv(k * DIM * 8)
            total_received_bytes += len(received_centroids)
            centroids = np.frombuffer(received_centroids, dtype=np.float64).reshape((k, DIM))

            # Count iteration time (per iteration)
            iteration_time = time.time() - start_time
            iteration_times.append(iteration_time)
            print(f"Iteration time: {iteration_time} seconds")

        # Compute and print final accuracy
        final_inertia = compute_inertia(data, centroids, clusters)
        max_inertia = compute_max_inertia(data)
        accuracy = compute_accuracy(final_inertia, max_inertia)
        print("Final Accuracy:", accuracy)

        # output cluster assignments
        output_cluster_assignments(data, clusters)

        # Plot clustered data
        plot_clusters(data, centroids, clusters)

        # Plot iteration times
        plt.figure()
        plt.plot(range(1, MAX_ITER + 1), iteration_times, marker='o')
        plt.title("Iteration Time")
        plt.xlabel("Iteration")
        plt.ylabel("Time (seconds)")
        plt.savefig("iteration_times.png")

        # Print out total time cost and communication cost for party 1 - Hospital A
        total_protocol_time = time.time() - protocol_start_time
        print(f"Total Protocol Time: {total_protocol_time} seconds")
        print(f"Total Data Sent: {total_sent_bytes} bytes")
        print(f"Total Data Received: {total_received_bytes} bytes")

        # Close connection
        conn.close()

    # Party 2 - Hospital B side
    elif party_id == 2:
        # Connect to party 1
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', port))
        print("Connected to party 1.")

        for _ in range(MAX_ITER):
            # Receive and deserialize clusters from party 1
            received_data = s.recv(4096)  # Buffer size
            total_received_bytes += len(received_data)
            clusters = pickle.loads(received_data)

            # Update centroids
            centroids = update_centroids(data, clusters)

            # Serialize and send updated centroids to party 1
            serialized_centroids = centroids.tobytes()
            s.send(serialized_centroids)
            total_sent_bytes += len(serialized_centroids)


        # Close connection.
        s.close()

    #  Print out total time cost and communication cost for party 2 - Hospital B     
        print(f"Total Protocol Time: {time.time() - protocol_start_time} seconds")
        print(f"Total Data Sent: {total_sent_bytes} bytes")
        print(f"Total Data Received: {total_received_bytes} bytes")

if __name__ == "__main__":
    main(sys.argv)