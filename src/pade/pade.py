import math
import numpy as np

def _get_tube_neighbours(data, ref_idx, tube_dimension, n):
    # Get the data of the reference sample.
    ref = data[ref_idx]

    # Compute the distances from the reference sample.
    distances = []
    for idx, _ in enumerate(data):
        # Skip the reference sample.
        if idx == ref_idx:
            continue
        
        # Euclidean distance along all dimensions, except the tube dimension.
        dist = 0
        for dim, _ in enumerate(ref):
            if dim != tube_dimension:
                dist += (data[idx][dim] - ref[dim])**2
        distances.append((idx, math.sqrt(dist)))
    
    # Sort by distances and return the indices of n closest neighbours.
    distances = sorted(distances, key=lambda x:x[1])
    return [i for (i, _) in distances[:n]]

def pade(data, target, nNeighbours=10):
    # Get the data dimension.
    (_, nAttributes) = data.shape
    
    # Initialize the Q-table with the same dimension as data.
    q_table = np.zeros(data.shape)

    # Make a tube regression along each dimension.
    for dim in range(nAttributes):
        # Process all samples along the current dimension.
        for idx, sample in enumerate(data):
            # Get the sample value.
            x0 = sample[dim]

            # Get the indices of the nearest neighbours within the tube.
            neighbours = _get_tube_neighbours(data, idx, dim, nNeighbours)

            # If not enough neighbours, skip this sample.
            if len(neighbours) < nNeighbours:
                continue
            
            # Take the target values of the returned neighbours.
            values = np.take(target, neighbours, axis=0)

            # Get the x values of the nearest neighbours.
            neighbours_x = np.take(data, neighbours, axis=0)[:,dim]
            
            # Compute the distance of the farthest neighbour.
            max_distance = max([abs(x - x0) for x in neighbours_x])

            # Compute the sigma parameter so that the weight of the farthest sample is 0.001.
            if max_distance < 1e-10:
                sg = math.log(.001)
            else:
                sg = math.log(.001) / max_distance**2
            
            # Compute the weighted univariate linear regression.
            Sx = Sy = Sxx = Syy = Sxy = n = 0.0
            for (x, y) in zip(neighbours_x, values):
                w = math.exp(sg * (x - x0)**2)
                Sx += w * x
                Sy += w * y
                Sxx += w * x**2
                Syy += w * y**2
                Sxy += w * x * y
                n += w
            div = n * Sxx - Sx**2
            if div != 0:
                b = (Sxy * n - Sx * Sy) / div
            else:
                b = 0

            # Store the sign of the partial derivative to the Q-table.
            q_table[idx][dim] = np.sign(b)

    # Return the Q-table.
    return q_table

# Translate the signs to the Q-notation.
def create_q_labels(q_table, attribute_names):
    labels = []
    
    for sample in q_table:
        label = 'Q('
        for name, value in zip(attribute_names, sample):
            if value > 0:
                if label != 'Q(':
                    label += ', '
                label += '+{}'.format(name)
            elif value < 0:
                if label != 'Q(':
                    label += ', '
                label += '-{}'.format(name)
        label += ')'
        labels.append(label)

    return np.array(labels)

# Translate the Q-labels to enumerated classes 0, 1, 2, ...
def enumerate_q_labels(q_labels):
    # The unique class names.
    class_names = np.unique(q_labels)
    
    # Enumerate classes.
    classes = np.array([np.where(class_names == q_label)[0][0] for q_label in q_labels])
    
    return classes, class_names