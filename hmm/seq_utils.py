import numpy as np
from sklearn.decomposition import PCA


def project_embeddings_and_centroids_together(embeddings, centroids, masks):

    original_embeddings_shape = embeddings.shape
    embeddings = np.reshape(embeddings, (-1, embeddings.shape[2]))
    points = np.concatenate([centroids, embeddings], axis=0)

    model = PCA(n_components=2)
    projected_points = model.fit_transform(points)

    projected_centroids = projected_points[:len(centroids)]
    projected_embeddings = projected_points[len(centroids):]
    projected_embeddings = np.reshape(
        projected_embeddings, (original_embeddings_shape[0], original_embeddings_shape[1], 2)
    )

    return projected_embeddings, projected_centroids


def project_embeddings(embeddings, masks, dimensionality):

    flat_masked_embeddings = mask_embeddings(embeddings, masks)

    model = PCA(n_components=dimensionality)
    model.fit(flat_masked_embeddings)
    projected_flat_embeddings = model.transform(flatten(embeddings))

    return unflatten(projected_flat_embeddings, embeddings.shape[1])


def mask_embeddings(embeddings, masks):

    flat_embeddings = flatten(embeddings)
    flat_masks = flatten(masks)

    flat_masked_embeddings = flat_embeddings[flat_masks]

    return flat_masked_embeddings


def flatten(seq):
    return np.reshape(seq, (seq.shape[0] * seq.shape[1], *seq.shape[2:]))


def unflatten(flat_seq, seq_length):
    return np.reshape(flat_seq, (-1, seq_length, *flat_seq.shape[1:]))
