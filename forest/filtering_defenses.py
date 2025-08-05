import torch
import numpy as np
import torch.nn as nn
import copy

from forest.utils import bypass_last_layer
from .data.datasets import PoisonSet, normalization
from torch.utils.data import Subset
from tqdm import tqdm
from .consts import NORMALIZE
from sklearn.decomposition import FastICA
from sklearn.metrics import silhouette_score
from .victims.models import get_model

def get_defense(defense):
    if defense == 'ss':
        return _SpectralSignature
    elif defense== 'deepknn':
        return _DeepKNN
    elif defense == 'ac':
        return _ActivationClustering
    elif defense == 'spectre':
        return _Spectre
    elif defense== 'scan':
        return _Scan
    elif defense == 'ct':
        return _CT
    else:
        raise NotImplementedError('Defense is not implemented')

def _get_poisoned_features(kettle, victim, poison_delta, dryrun=False):
    class_indices = [[] for _ in range(len(kettle.trainset_class_names))]
    feats = []
    layer_cake = list(victim.model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    feature_extractor.eval()
    with torch.no_grad():
        for i, (img, source, idx) in enumerate(kettle.trainset):
            lookup = kettle.poison_lookup.get(idx)
            if lookup is not None and poison_delta is not None:
                img += poison_delta[lookup, :, :, :]
            if NORMALIZE:
                img = normalization(img).to(**kettle.setup)
            else:
                img = img.unsqueeze(0).to(**kettle.setup)
            feats.append(feature_extractor(img))
            class_indices[source].append(i)
            # if dryrun and i == 3:  # use a few values to populate these adjancency matrices
            #     break
    return feats, class_indices

def cluster_metrics(cluster_1, cluster_0):

    num = len(cluster_1) + len(cluster_0)
    features = torch.cat([cluster_1, cluster_0], dim=0)

    labels = torch.zeros(num)
    labels[:len(cluster_1)] = 1
    labels[len(cluster_1):] = 0

    ## Raw Silhouette Score
    raw_silhouette_score = silhouette_score(features, labels)
    return raw_silhouette_score

def _get_cleaned_features(kettle, victim, dryrun=False):
    class_indices = [[] for _ in range(len(kettle.trainset_class_names))]
    feats = []
    layer_cake = list(victim.model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    feature_extractor.eval()
    with torch.no_grad():
        for i, (img, source, idx) in enumerate(kettle.validset):
            img = img.unsqueeze(0).to(**kettle.setup)
            feats.append(feature_extractor(img))
            class_indices[source].append(i)
            # if dryrun and i == 3:  # use a few values to populate these adjancency matrices
            #     break
    return feats, class_indices

    
def _DeepKNN(kettle, victim, poison_delta, args, num_classes=10, overestimation_factor=2.0):
    """deepKNN as in Peri et al. "Deep k-NN Defense against Clean-label Data Poisoning Attacks".

    An overestimation factor of 2 is motivated as necessary in that work."""
    clean_indices = []
    target_class = kettle.poison_setup['poison_class']
    num_poisons_expected = int(overestimation_factor * kettle.args.alpha * len(kettle.trainset_dist[target_class])) if not kettle.args.dryrun else 0
    feats, _ = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)

    feats = torch.stack(feats, dim=0)
    dist_matrix = torch.zeros((len(feats), len(feats)))
    for i in range(dist_matrix.shape[0]):
        temp_matrix = torch.stack([feats[i] for _ in range(dist_matrix.shape[1])], dim=0)
        dist_matrix[i, :] = torch.norm((temp_matrix - feats).squeeze(1), dim=1)
    for i in range(dist_matrix.shape[0]):
        vec = dist_matrix[i, :]
        point_label, _ = kettle.trainset.get_target(i)
        _, nearest_indices = vec.topk(num_poisons_expected + 1, largest=False)
        count = 0
        for j in range(1, num_poisons_expected + 1):
            neighbor_label, _ = kettle.trainset.get_target(nearest_indices[j])
            if neighbor_label == point_label:
                count += 1
            else:
                count -= 1
        if count >= 0:
            clean_indices.append(i)
    return clean_indices


def _SpectralSignature(kettle, victim, poison_delta, args, num_classes=10, overestimation_factor=1.5):
    """The spectral signature defense proposed by Tran et al. in "Spectral Signatures in Backdoor Attacks"

    https://proceedings.neurips.cc/paper/2018/file/280cf18baf4311c92aa5a042336587d3-Paper.pdf
    The overestimation factor of 1.5 is proposed in the paper.
    """
    clean_indices = []
    target_class = kettle.poison_setup['poison_class']
    num_poisons_expected = kettle.args.alpha * len(kettle.trainset_dist[target_class])
    feats, class_indices = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)

    for i in range(len(class_indices)):
        if len(class_indices[i]) > 1:
            temp_feats = []
            for temp_index in class_indices[i]:
                temp_feats.append(feats[temp_index])
            temp_feats = torch.cat(temp_feats)
            mean_feat = torch.mean(temp_feats, dim=0)
            temp_feats = temp_feats - mean_feat
            
            # Compute SVD
            U, S, Vh = torch.linalg.svd(temp_feats, full_matrices=False)
            
            # Use the first singular vector (corresponds to largest singular value)
            vec = Vh[0, :]  # First row of Vh
            
            vals = []
            for j in range(temp_feats.shape[0]):
                # Compute projection along the first right singular vector
                vals.append(torch.dot(temp_feats[j], vec).pow(2))

            k = min(int(overestimation_factor * num_poisons_expected), len(vals) - 1)
            _, indices = torch.topk(torch.tensor(vals), k)
            bad_indices = []
            for temp_index in indices:
                bad_indices.append(class_indices[i][temp_index])
            clean = list(set(class_indices[i]) - set(bad_indices))
            clean_indices = clean_indices + clean
    return clean_indices

def _ActivationClustering(kettle, victim, poison_delta, args, num_classes=10, clusters=2, threshold=0.1):
    """This is Chen et al. "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering" """
    # lazy sklearn import:
    from sklearn.cluster import KMeans

    suspicious_indices = []
    feats, class_indices = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)
    
    for target_class in range(len(class_indices)):
        if len(class_indices[target_class]) > 1:
            temp_feats = np.array([feats[temp_idx].squeeze(0).cpu().numpy() for temp_idx in class_indices[target_class]])
            
            ica = FastICA(n_components=10, max_iter=1000, tol=0.005)
            projected_feats = ica.fit_transform(temp_feats)
            kmeans = KMeans(n_clusters=clusters).fit(projected_feats)
            if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
                clean_label = 1
            else:
                clean_label = 0
                
            # by default, take the smaller cluster as the poisoned cluster
            if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
                clean_label = 1
            else:
                clean_label = 0

            outliers = []
            for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
                if bool:
                    outliers.append(class_indices[target_class][idx])

            score = silhouette_score(projected_feats, kmeans.labels_)
            print('[class-%d] silhouette_score = %f' % (target_class, score))
            
            if score > threshold and len(outliers) < len(kmeans.labels_) * 0.35: # if one of the two clusters is abnormally small
            # if len(outliers) < len(kmeans.labels_) * 0.35:
                print(f"Outlier Num in Class {target_class}:", len(outliers))
                suspicious_indices += outliers
    
    clean_indices = list(set(range(len(kettle.trainset))) - set(suspicious_indices))
    return clean_indices

def _CT(kettle, victim, poison_delta, args, num_classes=10):
    ct_detector = ConfusionTraining(kettle, victim, poison_delta, debug_info=True)
    clean_indices = ct_detector.detect()
    return clean_indices

def _Scan(kettle, victim, poison_delta, args, num_classes=10):
    # Use single-worker DataLoader to avoid multiprocessing bugs
    kwargs = {'num_workers': 0, 'pin_memory': True}

    inspection_set = PoisonSet(dataset=kettle.trainset, poison_delta=poison_delta, poison_lookup=kettle.poison_lookup, normalize=NORMALIZE)
    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=64, shuffle=False, **kwargs)

    clean_set_loader = torch.utils.data.DataLoader(
        kettle.validset,
        batch_size=64, shuffle=True, **kwargs)
    
    feats_inspection, class_indices_inspection = get_features(inspection_split_loader, victim.model)
    feats_clean, class_indices_clean = get_features(clean_set_loader, victim.model)

    # feats_inspection = np.array(feats_inspection)
    # class_indices_inspection = np.array(class_indices_inspection)
    # feats_clean = np.array(feats_clean)
    # class_indices_clean = np.array(class_indices_clean)

    # # Debug: Check for empty, nan, inf
    # for arr, name in [
    #     (feats_inspection, "feats_inspection"),
    #     (class_indices_inspection, "class_indices_inspection"),
    #     (feats_clean, "feats_clean"),
    #     (class_indices_clean, "class_indices_clean"),
    # ]:
    #     print(f"{name} shape={arr.shape}, nan={np.isnan(arr).any()}, inf={np.isinf(arr).any()}")

    # Initialize SCAn with CUDA support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scan = SCAn(device=device)

    print("Building global model...")
    # Convert features to numpy arrays for global model
    feats_clean_np = np.array(feats_clean)
    class_indices_clean_np = np.array(class_indices_clean)
    gb_model = scan.build_global_model(feats_clean_np, class_indices_clean_np, num_classes)
    
    size_inspection_set = len(feats_inspection)

    # Convert features to numpy arrays first, then concatenate
    feats_inspection_np = np.array(feats_inspection)
    feats_clean_np = np.array(feats_clean)
    class_indices_inspection_np = np.array(class_indices_inspection)
    class_indices_clean_np = np.array(class_indices_clean)
    
    feats_all = np.concatenate([feats_inspection_np, feats_clean_np])
    class_indices_all = np.concatenate([class_indices_inspection_np, class_indices_clean_np])

    print("Building local model...")
    lc_model = scan.build_local_model(feats_all, class_indices_all, gb_model, num_classes)
    score = scan.calc_final_score(lc_model)
    
    threshold = np.e  # This seems weird, but I kept your original logic
    print(f"Threshold for detection: {threshold}")
    suspicious_indices = []

    for target_class in range(num_classes):
        print(f'[class-{target_class}] outlier_score = {score[target_class]:.6f}')
        if score[target_class] <= threshold:
            continue

        tar_label = (class_indices_all == target_class)
        all_label = np.arange(len(class_indices_all))
        tar = all_label[tar_label]

        cluster_0_indices, cluster_1_indices = [], []
        cluster_0_clean, cluster_1_clean = [], []

        for index, i in enumerate(lc_model['subg'][target_class]):
            if i == 1:
                if tar[index] > size_inspection_set:
                    cluster_1_clean.append(tar[index])
                else:
                    cluster_1_indices.append(tar[index])
            else:
                if tar[index] > size_inspection_set:
                    cluster_0_clean.append(tar[index])
                else:
                    cluster_0_indices.append(tar[index])

        # decide which cluster is the poison cluster
        if len(cluster_0_clean) < len(cluster_1_clean):
            suspicious_indices += cluster_0_indices
        else:
            suspicious_indices += cluster_1_indices

    clean_idcs = list(set(range(len(kettle.trainset))) - set(suspicious_indices))
    return clean_idcs

EPS = 1e-5
class SCAn:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"SCAn initialized with device: {self.device}")

    def calc_final_score(self, lc_model=None):
        if lc_model is None:
            lc_model = self.lc_model
        sts = lc_model['sts']
        y = sts[:, 1]
        # Add small epsilon to avoid division by zero
        if isinstance(y, torch.Tensor):
            max_y = torch.max(y)
            max_y = max_y + 1e-8 if max_y == 0 else max_y
            ai = self.calc_anomaly_index(y / max_y)
            return ai.cpu().numpy() if isinstance(ai, torch.Tensor) else ai
        else:
            max_y = np.max(y)
            max_y = max_y + 1e-8 if max_y == 0 else max_y
            ai = self.calc_anomaly_index(y / max_y)
            return ai

    def calc_anomaly_index(self, a):
        if isinstance(a, torch.Tensor):
            ma = torch.median(a)
            b = torch.abs(a - ma)
            mm = torch.median(b) * 1.4826
            # Add small epsilon to avoid division by zero
            mm = mm + 1e-8 if mm == 0 else mm
            index = b / mm
            return index
        else:
            ma = np.median(a)
            b = abs(a - ma)
            mm = np.median(b) * 1.4826
            # Add small epsilon to avoid division by zero
            mm = mm + 1e-8 if mm == 0 else mm
            index = b / mm
            return index

    def build_global_model(self, reprs, labels, n_classes):
        # Convert inputs to PyTorch tensors if they aren't already
        if isinstance(reprs, np.ndarray):
            reprs = torch.from_numpy(reprs).float()
        elif isinstance(reprs, list):
            reprs = torch.tensor(reprs).float()
        else:
            reprs = reprs.float()
            
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()
        elif isinstance(labels, list):
            # Convert list to numpy first, handling scalar tensors and numpy scalars
            labels_array = []
            for label in labels:
                if isinstance(label, torch.Tensor):
                    labels_array.append(label.item())
                elif isinstance(label, (np.ndarray, np.integer)):
                    labels_array.append(int(label))
                else:
                    labels_array.append(int(label))
            labels = torch.tensor(labels_array, dtype=torch.long)
        else:
            labels = labels.long()

        N = reprs.shape[0]  # num_samples
        M = reprs.shape[1]  # len_features
        L = n_classes

        # Check if features are too large for GPU memory
        # Estimate memory usage: roughly N*M*4 bytes for features + M*M*4 bytes for covariance
        estimated_memory_gb = (N * M * 4 + M * M * 4) / (1024**3)
        use_cpu = estimated_memory_gb > 10 or M > 10000  # Use CPU if > 10GB or features > 10k
        
        if use_cpu:
            print(f"Using CPU for computation due to large feature size ({M}) or estimated memory usage ({estimated_memory_gb:.2f} GB)")
            device = torch.device('cpu')
        else:
            device = self.device
            
        reprs = reprs.to(device)
        labels = labels.to(device)

        # Safety checks
        if N == 0 or M == 0:
            raise ValueError("Empty features array provided")
        if torch.any(torch.isnan(reprs)) or torch.any(torch.isinf(reprs)):
            print("Warning: NaN or Inf values detected in representations, replacing with zeros")
            reprs = torch.nan_to_num(reprs, nan=0.0, posinf=0.0, neginf=0.0)

        mean_a = torch.mean(reprs, dim=0)
        X = reprs - mean_a

        # Vectorized computation of class means
        cnt_L = torch.zeros(L, device=device)
        mean_f = torch.zeros(L, M, device=device)
        
        for k in range(L):
            idx = (labels == k)
            cnt_L[k] = torch.sum(idx.float())
            if cnt_L[k] > 0:
                mean_f[k] = torch.mean(X[idx], dim=0)
            else:
                mean_f[k] = torch.zeros(M, device=device)

        # Vectorized computation of u and e
        u = mean_f[labels]  # Broadcasting to get class means for each sample
        e = X - u

        # Compute covariance matrices using PyTorch with memory-efficient approach
        try:
            # Try computation first
            Su = torch.cov(u.T)
            Se = torch.cov(e.T)
        except RuntimeError as e:
            # If runs out of memory, try with double precision or CPU
            print(f"Memory error during covariance computation: {e}")
            print("Trying CPU computation...")
            u_cpu = u.cpu()
            e_cpu = e.cpu()
            Su = torch.cov(u_cpu.T).to(device)
            Se = torch.cov(e_cpu.T).to(device)

        # Add regularization
        reg_eye = torch.eye(M, device=device) * 1e-8
        Su = Su + reg_eye
        Se = Se + reg_eye

        # EM iterations
        dist_Su = 1e5
        dist_Se = 1e5
        n_iters = 0
        
        while (dist_Su + dist_Se > EPS) and (n_iters < 100):
            n_iters += 1
            last_Su = Su.clone()
            last_Se = Se.clone()

            # Use more stable pseudo-inverse
            F = torch.linalg.pinv(Se, rcond=1e-8)
            SuF = torch.mm(Su, F)

            # Precompute G matrices for all classes
            G_set = []
            for k in range(L):
                matrix_to_invert = cnt_L[k] * Su + Se + reg_eye
                G = -torch.linalg.pinv(matrix_to_invert, rcond=1e-8)
                G = torch.mm(G, SuF)
                G_set.append(G)

            # Vectorized update of u_m
            u_m = torch.zeros(L, M, device=device)
            for i in range(N):
                vec = X[i]
                k = labels[i].item()
                G = G_set[k]
                dd = torch.mm(torch.mm(Se, G), vec.unsqueeze(1))
                u_m[k] = u_m[k] - dd.squeeze()

            # Update e and u
            u = u_m[labels]
            e = X - u

            # Recompute covariance matrices with memory-efficient approach
            try:
                Su = torch.cov(u.T) + reg_eye
                Se = torch.cov(e.T) + reg_eye
            except RuntimeError:
                # Fallback to CPU if runs out of memory
                u_cpu = u.cpu()
                e_cpu = e.cpu()
                Su = torch.cov(u_cpu.T).to(device) + reg_eye
                Se = torch.cov(e_cpu.T).to(device) + reg_eye

            # Compute convergence criteria
            dif_Su = Su - last_Su
            dif_Se = Se - last_Se

            dist_Su = torch.norm(dif_Su).item()
            dist_Se = torch.norm(dif_Se).item()

        # Convert back to numpy for compatibility with existing code
        gb_model = dict()
        gb_model['Su'] = Su.cpu().numpy()
        gb_model['Se'] = Se.cpu().numpy()
        gb_model['mean'] = mean_f.cpu().numpy()
        self.gb_model = gb_model
        return gb_model

    def build_local_model(self, reprs, labels, gb_model, n_classes):
        # Convert inputs to PyTorch tensors first without moving to GPU yet
        if isinstance(reprs, np.ndarray):
            reprs = torch.from_numpy(reprs).float()
        elif isinstance(reprs, list):
            reprs = torch.tensor(reprs).float()
        else:
            reprs = reprs.float()
            
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()
        elif isinstance(labels, list):
            # Convert list to numpy first, handling scalar tensors and numpy scalars
            labels_array = []
            for label in labels:
                if isinstance(label, torch.Tensor):
                    labels_array.append(label.item())
                elif isinstance(label, (np.ndarray, np.integer)):
                    labels_array.append(int(label))
                else:
                    labels_array.append(int(label))
            labels = torch.tensor(labels_array, dtype=torch.long)
        else:
            labels = labels.long()

        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        # Check if features are too large for GPU memory
        estimated_memory_gb = (N * M * 4 + M * M * 4) / (1024**3)
        use_cpu = estimated_memory_gb > 10 or M > 10000
        
        if use_cpu:
            print(f"Using CPU for local model due to large feature size ({M}) or estimated memory usage ({estimated_memory_gb:.2f} GB)")
            device = torch.device('cpu')
        else:
            device = self.device

        # Convert global model matrices to appropriate device
        if isinstance(gb_model['Su'], np.ndarray):
            Su = torch.from_numpy(gb_model['Su']).float().to(device)
        else:
            Su = gb_model['Su'].to(device)
            
        if isinstance(gb_model['Se'], np.ndarray):
            Se = torch.from_numpy(gb_model['Se']).float().to(device)
        else:
            Se = gb_model['Se'].to(device)

        # Move data to device
        reprs = reprs.to(device)
        labels = labels.to(device)

        F = torch.linalg.pinv(Se, rcond=1e-8)
        N = reprs.shape[0]
        M = reprs.shape[1]
        L = n_classes

        mean_a = torch.mean(reprs, dim=0)
        X = reprs - mean_a

        class_score = torch.zeros(L, 3, device=device)
        u1 = torch.zeros(L, M, device=device)
        u2 = torch.zeros(L, M, device=device)
        split_rst = []

        for k in range(L):
            selected_idx = (labels == k)
            cX = X[selected_idx]
            
            if cX.shape[0] == 0:  # Handle empty classes
                subg = torch.tensor([], device=device)
                i_u1 = torch.zeros(M, device=device)
                i_u2 = torch.zeros(M, device=device)
                i_sc = torch.tensor(0.0, device=device)
            else:
                subg, i_u1, i_u2 = self.find_split(cX, F, device)
                i_sc = self.calc_test(cX, Su, Se, F, subg, i_u1, i_u2, device)

            split_rst.append(subg.cpu().numpy())
            u1[k] = i_u1
            u2[k] = i_u2
            class_score[k] = torch.tensor([k, i_sc.item(), torch.sum(selected_idx.float()).item()], device=device)

        lc_model = dict()
        lc_model['sts'] = class_score.cpu().numpy()
        lc_model['mu1'] = u1.cpu().numpy()
        lc_model['mu2'] = u2.cpu().numpy()
        lc_model['subg'] = split_rst

        self.lc_model = lc_model
        return lc_model

    def find_split(self, X, F, device=None):
        if device is None:
            device = self.device
        N = X.shape[0]
        M = X.shape[1]
        
        # Safety check for empty input
        if N == 0:
            return (torch.tensor([], device=device), 
                   torch.zeros(M, device=device), 
                   torch.zeros(M, device=device))
            
        subg = torch.rand(N, device=device)

        if N == 1:
            subg[0] = 0
            return (subg, X.clone(), X.clone())

        # Ensure both clusters have at least one point initially
        if torch.sum(subg >= 0.5) == 0:
            subg[0] = 1
        if torch.sum(subg < 0.5) == 0:
            subg[0] = 0
        
        last_z1 = -torch.ones(N, device=device)

        # EM iterations
        steps = 0
        eps_tensor = torch.tensor(EPS, device=device)
        
        while (torch.norm(subg - last_z1) > eps_tensor) and \
              (torch.norm((1 - subg) - last_z1) > eps_tensor) and \
              (steps < 100):
            steps += 1
            last_z1 = subg.clone()

            # max-step: calc u1 and u2
            idx1 = (subg >= 0.5)
            idx2 = (subg < 0.5)
            
            if torch.sum(idx1) == 0 or torch.sum(idx2) == 0:
                break
                
            if torch.sum(idx1) == 1:
                u1 = X[idx1].squeeze(0)
            else:
                u1 = torch.mean(X[idx1], dim=0)
                
            if torch.sum(idx2) == 1:
                u2 = X[idx2].squeeze(0)
            else:
                u2 = torch.mean(X[idx2], dim=0)

            # Vectorized computation of bias and deltas
            bias = torch.mm(torch.mm(u1.unsqueeze(0), F), u1.unsqueeze(1)) - \
                   torch.mm(torch.mm(u2.unsqueeze(0), F), u2.unsqueeze(1))
            bias = bias.squeeze()
            
            e2 = u1 - u2
            deltas = torch.mm(torch.mm(X, F), e2.unsqueeze(1)).squeeze()
            
            # Update subg using vectorized operations
            subg = (bias - 2 * deltas < 0).float()

        return (subg, u1, u2)

    def calc_test(self, X, Su, Se, F, subg, u1, u2, device=None):
        if device is None:
            device = self.device
        N = X.shape[0]
        M = X.shape[1]

        # Add regularization to avoid singular matrices
        reg_eye = torch.eye(Se.shape[0], device=device) * 1e-8
        matrix_to_invert = N * Su + Se + reg_eye
        G = -torch.linalg.pinv(matrix_to_invert, rcond=1e-8)
        
        mu = torch.zeros(1, M, device=device)
        SeG = torch.mm(Se, G)
        
        # Vectorized computation of mu
        for i in range(N):
            vec = X[i]
            dd = torch.mm(SeG, vec.unsqueeze(1))
            mu = mu - dd.T

        # Compute bias terms
        b1 = torch.mm(torch.mm(mu, F), mu.T) - torch.mm(torch.mm(u1.unsqueeze(0), F), u1.unsqueeze(1))
        b2 = torch.mm(torch.mm(mu, F), mu.T) - torch.mm(torch.mm(u2.unsqueeze(0), F), u2.unsqueeze(1))
        
        n1 = torch.sum(subg >= 0.5).float()
        n2 = N - n1
        sc = n1 * b1 + n2 * b2

        # Vectorized computation of the correction term
        for i in range(N):
            e1 = X[i]
            if subg[i] >= 0.5:
                e2 = mu.squeeze() - u1
            else:
                e2 = mu.squeeze() - u2
            sc -= 2 * torch.mm(torch.mm(e1.unsqueeze(0), F), e2.unsqueeze(1))

        return sc / N

def get_features(data_loader, model):
    class_indices, feats = [], []

    layer_cake = list(model.children())
    feature_extractor = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
    feature_extractor.eval()
    model_device = next(model.parameters()).device
    feature_extractor = feature_extractor.to(model_device)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # Support both 2 or 3 returned items
            if len(batch) == 3:
                ins_data, ins_target, _ = batch
            elif len(batch) == 2:
                ins_data, ins_target = batch
            else:
                raise RuntimeError(f"Unexpected batch format: {type(batch)}, len={len(batch)}")
            ins_data = ins_data.to(model_device)
            x_features = feature_extractor(ins_data)
            for bid in range(ins_target.size(0)):
                feats.append(x_features[bid].cpu().numpy())
                class_indices.append(ins_target[bid].cpu().numpy())
    return feats, class_indices

def _Spectre(kettle, victim, poison_delta, args, num_classes=10):
    """
    Spectre defense method implementation.
    Returns a list of clean indices.
    """
    # Load data using your framework's data loading function
    feats, class_indices = _get_poisoned_features(kettle, victim, poison_delta, dryrun=kettle.args.dryrun)
    clean_feats, clean_class_indices = _get_cleaned_features(kettle, victim, dryrun=kettle.args.dryrun)
    suspicious_indices = []
    raw_poison_rate = args.alpha
    budget = int(raw_poison_rate * len(kettle.trainset_dist[kettle.poison_setup['poison_class']]) * 1.5)

    max_dim = 2 # 64
    class_taus = []
    class_S = []
    # Determine device from model
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    for i in range(num_classes):

        if len(class_indices[i]) > 1:

            # feats for class i in poisoned set
            temp_feats = np.array([feats[temp_idx].squeeze(0).cpu().numpy() for temp_idx in class_indices[i]])
            temp_feats = torch.FloatTensor(temp_feats).to(device)

            temp_clean_feats = None
            temp_clean_feats = np.array([clean_feats[temp_idx].squeeze(0).cpu().numpy() for temp_idx in clean_class_indices[i]])
            temp_clean_feats = torch.FloatTensor(temp_clean_feats).to(device)
            temp_clean_feats = temp_clean_feats - temp_feats.mean(dim=0)
            temp_clean_feats = temp_clean_feats.T

            temp_feats = temp_feats - temp_feats.mean(dim=0) # centered data
            temp_feats = temp_feats.T # feats arranged in column

            U, _, _ = torch.svd(temp_feats)
            U = U[:, :max_dim]

            # full projection
            projected_feats = torch.matmul(U.T, temp_feats)

            max_tau = -999999
            best_n_dim = -1
            best_to_be_removed = None

            for n_dim in range(2, max_dim+1): # enumarate all possible "reudced dimensions" and select the best

                S_removed, S_left = SPECTRE(U, temp_feats, n_dim, budget, temp_clean_feats)

                left_feats = projected_feats[:, S_left]
                covariance = torch.cov(left_feats)

                L, V = torch.linalg.eig(covariance)
                L, V = L.real, V.real
                L = (torch.diag(L) ** (1 / 2) + 0.001).inverse()
                normalizer = torch.matmul(V, torch.matmul(L, V.T))

                whitened_feats = torch.matmul(normalizer, projected_feats)

                tau = QUEscore(whitened_feats, max_dim).mean()

                if tau > max_tau:
                    max_tau = tau
                    best_n_dim = n_dim
                    best_to_be_removed = S_removed


            # print('class=%d, dim=%d, tau=%f' % (i, best_n_dim, max_tau))

            class_taus.append(max_tau)

            suspicious_indices = []
            for temp_index in best_to_be_removed:
                suspicious_indices.append(class_indices[i][temp_index])

            class_S.append(suspicious_indices)
        else:
            # Handle classes with <= 1 sample
            class_taus.append(0.0)  # Default tau value for classes with insufficient samples
            class_S.append([])  # Empty list for classes with insufficient samples

    class_taus = np.array(class_taus)
    median_tau = np.median(class_taus)

    #print('median_tau : %d' % median_tau)
    suspicious_indices = []
    max_tau = -99999
    for i in range(num_classes):
        #if class_taus[i] > max_tau:
        #    max_tau = class_taus[i]
        #    suspicious_indices = class_S[i]
        #print('class-%d, tau = %f' % (i, class_taus[i]))
        #if class_taus[i] > 2*median_tau:
        #    print('[large tau detected] potential poisons! Apply Filter!')
        for temp_index in class_S[i]:
            suspicious_indices.append(temp_index)
    clean_indices = list(set(range(len(kettle.trainset))) - set(suspicious_indices))
    return clean_indices

def QUEscore(temp_feats, n_dim):

    n_samples = temp_feats.shape[1]
    alpha = 4.0
    Sigma = torch.matmul(temp_feats, temp_feats.T) / n_samples
    I = torch.eye(n_dim).to(temp_feats.device)
    Q = torch.exp((alpha * (Sigma - I)) / (torch.linalg.norm(Sigma, ord=2) - 1))
    trace_Q = torch.trace(Q)

    taus = []
    for i in range(n_samples):
        h_i = temp_feats[:, i:i + 1]
        tau_i = torch.matmul(h_i.T, torch.matmul(Q, h_i)) / trace_Q
        tau_i = tau_i.item()
        taus.append(tau_i)
    taus = np.array(taus)

    return taus

def SPECTRE(U, temp_feats, n_dim, budget, oracle_clean_feats=None):

    projector = U[:, :n_dim].T # top left singular vectors
    temp_feats = torch.matmul(projector, temp_feats)

    if oracle_clean_feats is None:
        estimator = BeingRobust(random_state=0, keep_filtered=True).fit((temp_feats.T).cpu().numpy())
        clean_mean = torch.FloatTensor(estimator.location_).to(temp_feats.device)
        filtered_feats = (torch.FloatTensor(estimator.filtered_).to(temp_feats.device) - clean_mean).T
        clean_covariance = torch.cov(filtered_feats)
    else:
        clean_feats = torch.matmul(projector, oracle_clean_feats)
        clean_covariance = torch.cov(clean_feats)
        clean_mean = clean_feats.mean(dim = 1)


    temp_feats = (temp_feats.T - clean_mean).T

    # whiten the data
    L, V = torch.linalg.eig(clean_covariance)
    L, V = L.real, V.real
    L = (torch.diag(L)**(1/2)+0.001).inverse()
    normalizer = torch.matmul(V, torch.matmul( L, V.T ) )
    temp_feats = torch.matmul(normalizer, temp_feats)

    # compute QUEscore
    taus = QUEscore(temp_feats, n_dim)

    sorted_indices = np.argsort(taus)
    n_samples = len(sorted_indices)

    budget = min(budget, n_samples//2) # default assumption : at least a half of samples in each class is clean

    suspicious = sorted_indices[-budget:]
    left = sorted_indices[:n_samples-budget]

    return suspicious, left

from typing import Tuple, Union

import numpy as np
from scipy.special import erfc
from sklearn.utils.extmath import randomized_svd
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils import check_random_state


class BeingRobust(EmpiricalCovariance):
    """Being Robust (in High Dimensions) Can Be Practical: robust estimator of location (and potentially covariance).
    This estimator is to be applied on Gaussian-distributed data. For other distributions some changes might be
    required. Please check out the original paper and/or Matlab code.
    Parameters
    ----------
    eps : float, optional
        Fraction of perturbed data points, by default 0.1
    tau : float, optional
        Significance level, by default 0.1
    cher : float, optional
        Factor filter criterion, by default 2.5
    use_randomized_svd : bool, optional
        If True use `sklearn.utils.extmath.randomized_svd`, else use full SVD, by default True
    debug : bool, optional
        If True print debug information, by default False
    assume_centered : bool
        If True, the data is not centered beforehand, by default False
    random_state : Union[int, np.random.RandomState],
        Determines the pseudo random number generator for shuffling the data.
        Pass an int for reproducible results across multiple function calls. By default none
    keep_filtered : bool, optional
        If True teh filtered data point are kept (`BeingRobust#filtered_`) the, by default False
    Attributes
    ----------
    location_ : np.ndarray of shape (n_features,)
        Estimated robust location.
    filtered_ : np.ndarray of shape (?, n_features)
        Remaining data points for estimating the mean.
    Examples
    --------
    #>>> import numpy as np
    #>>> from being_robust import BeingRobust
    #>>> real_cov = np.array([[.8, .3], [.3, .4]])
    #>>> rng = np.random.RandomState(0)
    #>>> X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=500)
    #>>> br = BeingRobust(random_state=0, keep_filtered=True).fit(X)
    #>>> br.location_
    #array([0.0622..., 0.0193...])
    #>>> br.filtered_
    #array([[-1.6167..., -0.6431...], ...
    """

    def __init__(self,
                 eps: float = 0.1,
                 tau: float = 0.1,
                 cher: float = 2.5,
                 use_randomized_svd: bool = True,
                 debug: bool = False,
                 assume_centered: bool = False,
                 random_state: Union[int, np.random.RandomState] = None,
                 keep_filtered: bool = False):
        super().__init__()
        self.eps = eps
        self.tau = tau
        self.cher = cher
        self.use_randomized_svd = use_randomized_svd
        self.debug = debug
        self.random_state = random_state
        self.assume_centered = assume_centered
        self.keep_filtered = keep_filtered

    def fit(self, X, y=None) -> 'BeingRobust':
        """Fits the data to obtain the robust estimate.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y: Ignored
            Not used, present for API consistence purpose.
        Returns
        -------
        self : BeingRobust
        """
        X = self._validate_data(X, ensure_min_samples=1, estimator='BeingRobust')
        random_state = check_random_state(self.random_state)

        self.location_, X = filter_gaussian_mean(X,
                                                 eps=self.eps,
                                                 tau=self.tau,
                                                 cher=self.cher,
                                                 use_randomized_svd=self.use_randomized_svd,
                                                 debug=self.debug,
                                                 assume_centered=self.assume_centered,
                                                 random_state=random_state)
        if self.keep_filtered:
            self.filtered_ = X

        return self


def filter_gaussian_mean(X: np.ndarray,
                         eps: float = 0.1,
                         tau: float = 0.1,
                         cher: float = 2.5,
                         use_randomized_svd: bool = True,
                         debug: bool = False,
                         assume_centered: bool = False,
                         random_state: int = None) -> Tuple[float, np.ndarray]:
    """Being Robust (in High Dimensions) Can Be Practical: robust estimator of location (and potentially covariance).
    This estimator is to be applied on Gaussian-distributed data. For other distributions some changes might be
    required. Please check out the original paper and/or Matlab code.
    Parameters
    ----------
    eps : float, optional
        Fraction of perturbed data points, by default 0.1
    tau : float, optional
        Significance level, by default 0.1
    cher : float, optional
        Factor filter criterion, by default 2.5
    use_randomized_svd : bool, optional
        If True use `sklearn.utils.extmath.randomized_svd`, else use full SVD, by default True
    debug : bool, optional
        If True print debug information, by default False
    assume_centered : bool
        If True, the data is not centered beforehand, by default False
    random_state : Union[int, np.random.RandomState],
        Determines the pseudo random number generator for shuffling the data.
        Pass an int for reproducible results across multiple function calls. By default none
    Returns
    -------
    Tuple[float, np.ndarray]
        The robust location estimate, the filtered version of `X`
    """
    n_samples, n_features = X.shape

    emp_mean = X.mean(axis=0)

    if assume_centered:
        centered_X = X
    else:
        centered_X = (X - emp_mean) / np.sqrt(n_samples)

    if use_randomized_svd:
        U, S, Vh = randomized_svd(centered_X.T, n_components=1, random_state=random_state)
    else:
        U, S, Vh = np.linalg.svd(centered_X.T, full_matrices=False)

    lambda_ = S[0]**2
    v = U[:, 0]

    if debug:
        print(f'\nRecursing on X of shape {X.shape}')
        print(f'lambda_ < 1 + 3 * eps * np.log(1 / eps) -> {lambda_} < {1 + 3 * eps * np.log(1 / eps)}')
    if lambda_ < 1 + 3 * eps * np.log(1 / eps):
        return emp_mean, X

    delta = 2 * eps
    if debug:
        print(f'delta={delta}')

    projected_X = X @ v
    med = np.median(projected_X)
    projected_X = np.abs(projected_X - med)
    sorted_projected_X_idx = np.argsort(projected_X)
    sorted_projected_X = projected_X[sorted_projected_X_idx]

    for i in range(n_samples):
        T = sorted_projected_X[i] - delta
        filter_crit_lhs = n_samples - i
        filter_crit_rhs = cher * n_samples * \
            erfc(T / np.sqrt(2)) / 2 + eps / (n_samples * np.log(n_samples * eps / tau))
        if filter_crit_lhs > filter_crit_rhs:
            break

    if debug:
        print(f'filter data at index {i}')

    if i == 0 or i == n_samples - 1:
        return emp_mean, X

    return filter_gaussian_mean(
        X[sorted_projected_X_idx[:i + 1]],
        eps=eps,
        tau=tau,
        cher=cher,
        use_randomized_svd=use_randomized_svd,
        debug=debug,
        assume_centered=assume_centered,
        random_state=random_state
    )

class ConfusionTraining:
    def __init__(self, kettle, victim, poison_delta, debug_info=False, defense_ratio=0.2):
        self.kettle = kettle

        # Ensure all models are on cuda:0 before DataParallel wrapping
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        # Move victim model to the primary device and ensure all parameters are there
        self.base_model = victim.model
        if hasattr(self.base_model, 'module'):  # If already wrapped in DataParallel
            self.base_model = self.base_model.module  # Extract the underlying model
        self.base_model = self.base_model.to(device)
        
        # Create pretrained model and move to primary device
        self.pretrained_model = get_model(kettle.args.net[0], num_classes=kettle.num_classes, pretrained=True)
        self.pretrained_model = self.pretrained_model.to(device)
        
        # Create confused model as a copy of pretrained model
        self.confused_model = copy.deepcopy(self.pretrained_model)
        self.confused_model = self.confused_model.to(device)

        # Only wrap with DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.base_model = nn.DataParallel(self.base_model)
            self.pretrained_model = nn.DataParallel(self.pretrained_model)
            self.confused_model = nn.DataParallel(self.confused_model)

        self.debug_info = debug_info
        self.defense_ratio = defense_ratio

        self.distillation_ratio = [1/2, 1/5, 1/25, 1/50, 1/100]
        self.momentums = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        self.lambs = [20, 20, 20, 30, 30, 15]
        self.lrs = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01]
        self.batch_factors = [2, 2, 2, 2, 2, 2]

        self.weight_decay = 1e-4
        self.median_sample_rate = 0.1
        
        size = int(defense_ratio * len(self.kettle.validset))
        random_indices = torch.randperm(len(self.kettle.validset))[:size]
        self.clean_set = Subset(self.kettle.validset, indices=random_indices)
        self.clean_loader = torch.utils.data.DataLoader(self.clean_set, 
                                                        batch_size=64, 
                                                        shuffle=False, 
                                                        num_workers=4, 
                                                        pin_memory=True
                                                    )
        self.inspection_set = PoisonSet(dataset=kettle.trainset, poison_delta=poison_delta, poison_lookup=kettle.poison_lookup, normalize=NORMALIZE)

    def detect(self):
        distilled_samples_indices, median_sample_indices, model = self.iterative_poison_distillation(self.inspection_set,
                                        self.clean_set, self.debug_info)
        
        suspicious_indices = self.identify_poison_samples_simplified(self.inspection_set, median_sample_indices,
                                                                model)
        clean_indices = sorted(list(set(range(len(self.kettle.trainset))) - set(suspicious_indices)))
        return clean_indices
        

    def iterative_poison_distillation(self, inspection_set, clean_set, debug_info=None):
        clean_set_loader = torch.utils.data.DataLoader(clean_set, 
                                                batch_size=64,
                                                shuffle=True, 
                                                pin_memory=True,
                                                num_workers=4)

        print('>>> Iterative Data Distillation with Confusion Training')

        distilled_samples_indices, median_sample_indices = None, None
        num_confusion_iter = len(self.distillation_ratio)
        criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
        criterion = nn.CrossEntropyLoss()
           
        distilled_set = inspection_set
        for confusion_iter in range(num_confusion_iter):
            size_of_distilled_set = len(distilled_set)
            print('<Round-%d> Size_of_distillation_set = ' % confusion_iter, size_of_distilled_set)

            # different weights for each class based on their frequencies in the distilled set
            nums_of_each_class = np.zeros(self.kettle.num_classes)
            for i in range(size_of_distilled_set):
                _, gt, _ = distilled_set[i]
                if isinstance(gt, torch.Tensor):
                    gt = gt.item()
                nums_of_each_class[gt] += 1
            print(nums_of_each_class)
            freq_of_each_class = nums_of_each_class / size_of_distilled_set
            freq_of_each_class = np.sqrt(freq_of_each_class + 0.001)

            pretrain_epochs = 10
            pretrain_lr = 0.001
            distillation_iters = 100

            if confusion_iter == num_confusion_iter - 1:
                freq_of_each_class[:] = 1

            if confusion_iter != num_confusion_iter - 1:
                distilled_set_loader = torch.utils.data.DataLoader(
                    distilled_set,
                    batch_size=64,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=4)
            else:
                distilled_set_loader = torch.utils.data.DataLoader(
                    torch.utils.data.ConcatDataset([distilled_set, clean_set]),
                    batch_size=64,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=4)

            print('freq: ', freq_of_each_class)

            distilled_set_loader = torch.utils.data.DataLoader(
                distilled_set,
                batch_size=64, 
                shuffle=True,
                pin_memory=True,
                num_workers=4
            )
            self.pretrained_model = self.pretrain(confusion_iter=confusion_iter,
                        pretrain_model=self.pretrained_model, 
                        weight_decay=self.weight_decay, 
                        pretrain_epochs=pretrain_epochs, 
                        distilled_set_loader=distilled_set_loader, 
                        lr=pretrain_lr,
                        criterion=criterion)

            distilled_set_loader = torch.utils.data.DataLoader(
                distilled_set,
                batch_size=64, 
                shuffle=True,
                pin_memory=True,
                num_workers=4
            )
            
            # confusion_training
            # Create confused model with proper device handling
            if hasattr(self.pretrained_model, 'module'):
                # If wrapped in DataParallel, extract the underlying model first
                base_pretrained = self.pretrained_model.module
            else:
                base_pretrained = self.pretrained_model
            
            self.confused_model = copy.deepcopy(base_pretrained)
            
            # Ensure confused model is on the correct device
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            self.confused_model = self.confused_model.to(device)
            
            # Wrap with DataParallel if multiple GPUs are available
            if torch.cuda.device_count() > 1:
                self.confused_model = nn.DataParallel(self.confused_model)
            model = self.confusion_train(confusion_iter=confusion_iter, base_model=self.base_model, 
                                         confused_model=self.confused_model, 
                                         distilled_set_loader=distilled_set_loader, 
                                         clean_set_loader=clean_set_loader, 
                                         weight_decay=self.weight_decay, 
                                         criterion_no_reduction=criterion_no_reduction,
                                         momentum=self.momentums[confusion_iter], 
                                         lamb=self.lambs[confusion_iter],
                                         freq=freq_of_each_class, 
                                         lr=self.lrs[confusion_iter], 
                                         batch_factor=self.batch_factors[confusion_iter], 
                                         distillation_iters=distillation_iters)

            # distill the inspected set according to the loss values
            distilled_samples_indices, median_sample_indices = self.distill(self.confused_model, inspection_set,
                                                                                        confusion_iter, criterion_no_reduction)

            distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)

        return distilled_samples_indices, median_sample_indices, model
    

    def pretrain(self, confusion_iter, pretrain_model, weight_decay, pretrain_epochs, distilled_set_loader, lr, criterion=torch.nn.CrossEntropyLoss()):
        '''
            pretraining on the poisoned dataset to learn a prior of the backdoor

            Parameters:
                confusion_iter (int): the round id of the confusion training
                pretrain_model: the model to be pretrained
                weight_decay (float): weight_decay parameter for optimizer
                pretrain_epochs (int): number of pretraining epochs
                distilled_set_loader (torch.utils.data.DataLoader): data loader of the distilled set
                criterion: loss function
                lr (float): learning rate
    
            Returns:
                model: the pretrained model
        '''

        ######### Pretrain Base Model ##############
        optimizer = torch.optim.SGD(pretrain_model.parameters(), lr,  momentum=0.9, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4], gamma=0.1)

        for epoch in range(1, pretrain_epochs + 1):  # pretrain backdoored base model with the distilled set
            pretrain_model.train()

            for batch_idx, (data, target, idxs) in enumerate( tqdm(distilled_set_loader) ):
                optimizer.zero_grad()
                
                # Get device from the model
                if hasattr(pretrain_model, 'module'):
                    model_device = next(pretrain_model.module.parameters()).device
                else:
                    model_device = next(pretrain_model.parameters()).device
                
                data, target = data.to(model_device), target.to(model_device)  # train set batch
                output = pretrain_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()

            if epoch % 2 == 0:
                print('<Round-{} : Pretrain> Train Epoch: {}/{} \tLoss: {:.6f}'.format(confusion_iter, epoch, pretrain_epochs, loss.item()))
                if self.debug_info:
                    eval_model(pretrain_model, self.kettle)

        return pretrain_model
    
    def confusion_train(self, confusion_iter, base_model, confused_model, distilled_set_loader, clean_set_loader, 
                        weight_decay, criterion_no_reduction, momentum, lamb, freq, lr, batch_factor, distillation_iters):
        '''
        key codes for confusion training loop

        Parameters:
            confusion_iter (int): the round id of the confusion training
            base_model: the backdoored model
            confused_model: the confused model
            distilled_set_loader (torch.utils.data.DataLoader): the data loader of distilled set in previous rounds
            clean_set_loader (torch.utils.data.DataLoader): the data loader of the reserved clean set
            lamb (int): the weight parameter to balance confusion training objective and clean training objective
            freq (int): class frequency of the distilled set
            lr (float): learning rate of the optimizer
            batch_factor (int): the number of batch intervals of applying confusion training objective
            distillation_iters (int): the number of confusion training iterations

        Returns:
            the confused model in this round
        '''
        base_model.eval()

        ######### Distillation Step ################

        optimizer = torch.optim.SGD(confused_model.parameters(), lr=lr, weight_decay=weight_decay,
                                    momentum=momentum)

        distilled_set_iters = iter(distilled_set_loader)
        clean_set_iters = iter(clean_set_loader)


        rounder = 0
        for batch_idx in tqdm(range(distillation_iters)):

            try:
                data_shift, target_shift, idxs = next(clean_set_iters)
            except Exception as e:
                clean_set_iters = iter(clean_set_loader)
                data_shift, target_shift, idxs = next(clean_set_iters)

            # Get the device from the model
            if hasattr(base_model, 'module'):
                model_device = next(base_model.module.parameters()).device
            else:
                model_device = next(base_model.parameters()).device
            
            data_shift, target_shift = data_shift.to(model_device), target_shift.to(model_device)

            with torch.no_grad():
                preds = torch.argmax(base_model(data_shift), dim=1).detach()
                if (rounder + batch_idx) % self.kettle.num_classes == 0:
                    rounder += 1
                next_target = (preds + rounder + batch_idx) % self.kettle.num_classes
                target_confusion = next_target

            confused_model.train()

            if batch_idx % batch_factor == 0:
                try:
                    data, target, idxs = next(distilled_set_iters)
                except Exception as e:
                    distilled_set_iters = iter(distilled_set_loader)
                    data, target, idxs = next(distilled_set_iters)

                data, target = data.to(model_device), target.to(model_device)
                data_mix = torch.cat([data_shift, data], dim=0)
                target_mix = torch.cat([target_confusion, target], dim=0)
                boundary = data_shift.shape[0]

                output_mix = confused_model(data_mix)
                loss_mix = criterion_no_reduction(output_mix, target_mix)

                loss_inspection_batch_all = loss_mix[boundary:]
                loss_confusion_batch_all = loss_mix[:boundary]
                loss_confusion_batch = loss_confusion_batch_all.mean()
                target_inspection_batch_all = target_mix[boundary:]
                inspection_batch_size = len(loss_inspection_batch_all)
                loss_inspection_batch = 0
                normalizer = 0
                for i in range(inspection_batch_size):
                    gt = int(target_inspection_batch_all[i].item())
                    loss_inspection_batch += (loss_inspection_batch_all[i] / freq[gt])
                    normalizer += (1 / freq[gt])
                loss_inspection_batch = loss_inspection_batch / normalizer

                weighted_loss = (loss_confusion_batch * (lamb-1) + loss_inspection_batch) / lamb

                loss_confusion_batch = loss_confusion_batch.item()
                loss_inspection_batch = loss_inspection_batch.item()
            else:
                output = confused_model(data_shift)
                weighted_loss = loss_confusion_batch = criterion_no_reduction(output, target_confusion).mean()
                loss_confusion_batch = loss_confusion_batch.item()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print('<Round-{} : Distillation Step> Batch_idx: {}, batch_factor: {}, lr: {}, lamb : {}, moment : {}, Loss: {:.6f}'.format(
                    confusion_iter, batch_idx + 1, batch_factor, optimizer.param_groups[0]['lr'], lamb, momentum,
                    weighted_loss.item()))
                print('inspection_batch_loss = %f, confusion_batch_loss = %f' %
                    (loss_inspection_batch, loss_confusion_batch))

                if self.debug_info:
                    eval_model(confused_model, self.kettle)

        return confused_model

    def distill(self, confused_model, inspection_set, n_iter, criterion_no_reduction, final_budget = None, class_wise = False):
        '''
        distill samples from the dataset based on loss values of the inference model

        Parameters:
            confused_model: the model that is confused by Confusion Training
            inspection_set (torch.utils.data.Dataset): the dataset that potentially contains poison samples and needs to be cleansed
            n_iter (int): id of current iteration
            criterion_no_reduction: loss function
            final_budget (int): maximal number of distilled samples
            class_wise (bool): whether to list indices of distilled samples for each class seperately

        Returns:
            indicies of distilled samples
        '''
        num_samples = len(inspection_set)
        num_confusion_iter = len(self.distillation_ratio) + 1

        inspection_set_loader = torch.utils.data.DataLoader(
                                                    inspection_set, 
                                                    batch_size=256,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    pin_memory=True
                                                )

        """
            Collect loss values for inspected samples.
        """
        loss_array = []
        correct_instances = [] # Instances that the confused model correctly predicts
        gts = [] # Ground truth labels
        confused_model.eval()
        st = 0
        with torch.no_grad():

            for data, target, idxs in tqdm(inspection_set_loader):
                device = next(confused_model.parameters()).device
                data, target = data.to(device), target.to(device)
                output = confused_model(data)

                preds = torch.argmax(output, dim=1)
                batch_loss = criterion_no_reduction(output, target)
                this_batch_size = len(target)

                for i in range(this_batch_size):
                    loss_array.append(batch_loss[i].item())
                    gts.append(int(target[i].item()))

                    if preds[i] == target[i]:
                        correct_instances.append(st + i)

                st += this_batch_size

        loss_array = np.array(loss_array)
        sorted_indices = np.argsort(loss_array)
 
        top_indices_each_class = [[] for _ in range(self.kettle.num_classes)]
        for t in sorted_indices:
            gt = gts[t]
            top_indices_each_class[gt].append(t)

        """
            Distill samples with low loss values from the inspected set.
        """

        if n_iter < num_confusion_iter - 1:

            if self.distillation_ratio[n_iter] is None:
                distilled_samples_indices = head = correct_instances
            else:
                num_expected = int(self.distillation_ratio[n_iter] * num_samples)
                head = sorted_indices[:num_expected]
                head = list(head)
                distilled_samples_indices = head

            if n_iter < num_confusion_iter - 2: 
                rate_factor = 50
            else: 
                rate_factor = 100

            class_dist = np.zeros(self.kettle.num_classes, dtype=int)
            for i in distilled_samples_indices:
                gt = gts[i]
                class_dist[gt] += 1

            for i in range(self.kettle.num_classes):
                minimal_sample_num = len(top_indices_each_class[i]) // rate_factor
                print('class-%d, collected=%d, minimal_to_collect=%d' % (i, class_dist[i], minimal_sample_num) )
                if class_dist[i] < minimal_sample_num:
                    for k in range(class_dist[i], minimal_sample_num):
                        distilled_samples_indices.append(top_indices_each_class[i][k])

        else:
            if final_budget is not None:
                head = sorted_indices[:final_budget]
                head = list(head)
                distilled_samples_indices = head
            else:
                distilled_samples_indices = head = correct_instances

        distilled_samples_indices.sort()


        median_sample_indices = []
        sorted_indices_each_class = [[] for _ in range(self.kettle.num_classes)]
        for temp_id in sorted_indices:
            gt = gts[temp_id]
            sorted_indices_each_class[gt].append(temp_id)

        for i in range(self.kettle.num_classes):
            num_class_i = len(sorted_indices_each_class[i])
            st = int(num_class_i / 2 - num_class_i * self.median_sample_rate / 2)
            ed = int(num_class_i / 2 + num_class_i * self.median_sample_rate / 2)
            for temp_id in range(st, ed):
                median_sample_indices.append(sorted_indices_each_class[i][temp_id])

        """Report statistics of the distillation results...
        """
        if self.debug_info:

            print('num_correct : ', len(correct_instances))
            poison_indices = self.kettle.poison_target_ids

            cnt = 0
            for s, cid in enumerate(head):  # enumerate the head part
                original_id = cid
                if original_id in poison_indices:
                    cnt += 1

            print('How Many Poison Samples are Concentrated in the Head? --- %d/%d = %f' % (cnt, len(poison_indices), cnt/len(poison_indices)))

            poison_dist = []

            for temp_id in range(num_samples):
                if sorted_indices[temp_id] in poison_indices:
                    poison_dist.append(temp_id)

            print('poison distribution : ', poison_dist)

            num_poison = len(poison_indices)
            num_collected = len(correct_instances)
            pt = 0

            recall = 0
            for idx in correct_instances:
                if pt >= num_poison:
                    break
                while (idx > poison_indices[pt] and pt + 1 < num_poison): 
                    pt += 1
                if pt < num_poison and poison_indices[pt] == idx:
                    recall += 1

            fpr = num_collected - recall
            print('recall = %d/%d = %f, fpr = %d/%d = %f' % (recall, num_poison, recall/num_poison if num_poison!=0 else 0,
                                                                fpr, num_samples - num_poison,
                                                                fpr / (num_samples - num_poison) if (num_samples-num_poison)!=0 else 0))

        if class_wise:
            return distilled_samples_indices, median_sample_indices, top_indices_each_class
        else:
            return distilled_samples_indices, median_sample_indices

    def get_features(self, data_loader, model):
        '''
            Extract features on a dataset with a given model

            Parameters:
                data_loader (torch.utils.data.DataLoader): the dataloader of the dataset on which we want to extract features
                model (nn.Module): the mode used to extract features

            Returns:
                feats(list): a list of features for each sample in the dataset
                label_list(list): the ground truth label for each sample in the dataset
                preds_list(list): the model's prediction on each sample of the dataset
                gt_confidence(list): the model's confidence on the ground truth label of each sample in the dataset
                loss_vals(list): the loss values of the model on each sample in the dataset
            '''

        headless_model, last_layer = bypass_last_layer(model)
        label_list = []
        preds_list = []
        feats = []
        gt_confidence = []
        loss_vals = []

        criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
        model.eval()

        with torch.no_grad():

            for i, (ins_data, ins_target, _) in enumerate(tqdm(data_loader)):

                # Get device from the model
                if hasattr(model, 'module'):
                    model_device = next(model.module.parameters()).device
                else:
                    model_device = next(model.parameters()).device
                
                ins_data, ins_target = ins_data.to(model_device), ins_target.to(model_device)
                x_features = headless_model(ins_data)
                output = last_layer(x_features)

                loss = criterion_no_reduction(output, ins_target).cpu().numpy()

                preds = torch.argmax(output, dim=1).cpu().numpy()
                prob = torch.softmax(output, dim=1).cpu().numpy()
                this_batch_size = len(ins_target)

                for bid in range(this_batch_size):
                    gt = ins_target[bid].cpu().item()
                    feats.append(x_features[bid].cpu().numpy())
                    label_list.append(gt)
                    preds_list.append(preds[bid])
                    gt_confidence.append(prob[bid][gt])
                    loss_vals.append(loss[bid])
        return feats, label_list, preds_list, gt_confidence, loss_vals

    def identify_poison_samples_simplified(self, inspection_set, clean_indices, model):
        '''
        Identify poison samples in a dataset (under inspection) with the confused model.

        Parameters:
            inspection_set (torch.utils.data.Dataset): the dataset that potentially contains poison samples and needs to be cleansed
            clean_indices (dict): a set of indices of samples that are expected to be clean (e.g., have high loss values after confusion training)
            model (nn.Module): the model used to detect poison samples

        Returns:
            suspicious_indices (list): indices of detected poison samples
        '''

        from scipy.stats import multivariate_normal


        num_samples = len(inspection_set)

        # main dataset we aim to cleanse
        inspection_split_loader = torch.utils.data.DataLoader(
            inspection_set,
            batch_size=128, 
            shuffle=False, 
            pin_memory=True,
            num_workers=4,
        )

        model.eval()
        feats_inspection, class_labels_inspection, preds_inspection, \
        gt_confidence_inspection, loss_vals = self.get_features(inspection_split_loader, model)

        feats_inspection = np.array(feats_inspection)
        class_labels_inspection = np.array(class_labels_inspection)

        class_indices = [[] for _ in range(self.kettle.num_classes)]
        class_indices_in_clean_chunklet = [[] for _ in range(self.kettle.num_classes)]

        for i in range(num_samples):
            gt = int(class_labels_inspection[i])
            class_indices[gt].append(i)

        for i in clean_indices:
            gt = int(class_labels_inspection[i])
            class_indices_in_clean_chunklet[gt].append(i)

        for i in range(self.kettle.num_classes):
            class_indices[i].sort()
            class_indices_in_clean_chunklet[i].sort()

            if len(class_indices[i]) < 2:
                raise Exception('dataset is too small for class %d' % i)

            if len(class_indices_in_clean_chunklet[i]) < 2:
                raise Exception('clean chunklet is too small for class %d' % i)

        # apply cleanser, if the likelihood of two-clusters-model is twice of the likelihood of single-cluster-model
        threshold = 2
        suspicious_indices = []
        class_likelihood_ratio = []

        for target_class in range(self.kettle.num_classes):

            num_samples_within_class = len(class_indices[target_class])
            print('class-%d : ' % target_class, num_samples_within_class)
            clean_chunklet_size = len(class_indices_in_clean_chunklet[target_class])
            clean_chunklet_indices_within_class = []
            pt = 0
            for i in range(num_samples_within_class):
                if pt == clean_chunklet_size:
                    break
                if class_indices[target_class][i] < class_indices_in_clean_chunklet[target_class][pt]:
                    continue
                else:
                    clean_chunklet_indices_within_class.append(i)
                    pt += 1

            print('start_pca..')

            temp_feats = torch.FloatTensor(
                feats_inspection[class_indices[target_class]])
            
            # Determine device - use CPU for PCA operations as it's more stable
            device = torch.device('cpu')
            temp_feats = temp_feats.to(device)


            # reduce dimensionality
            U, S, V = torch.pca_lowrank(temp_feats, q=2)
            projected_feats = torch.matmul(temp_feats, V[:, :2]).cpu()

            # isolate samples via the confused inference model
            isolated_indices_global = []
            isolated_indices_local = []
            other_indices_local = []
            labels = []
            for pt, i in enumerate(class_indices[target_class]):
                if preds_inspection[i] == target_class:
                    isolated_indices_global.append(i)
                    isolated_indices_local.append(pt)
                    labels.append(1) # suspected as positive
                else:
                    other_indices_local.append(pt)
                    labels.append(0)

            projected_feats_isolated = projected_feats[isolated_indices_local]
            projected_feats_other = projected_feats[other_indices_local]

            print('========')
            print('num_isolated:', projected_feats_isolated.shape)
            print('num_other:', projected_feats_other.shape)

            num_isolated = projected_feats_isolated.shape[0]

            print('num_isolated : ', num_isolated)

            if (num_isolated >= 2) and (num_isolated <= num_samples_within_class - 2):

                mu = np.zeros((2,2))
                covariance = np.zeros((2,2,2))

                mu[0] = projected_feats_other.mean(axis=0)
                covariance[0] = np.cov(projected_feats_other.T)
                mu[1] = projected_feats_isolated.mean(axis=0)
                covariance[1] = np.cov(projected_feats_isolated.T)

                # avoid singularity
                covariance += 0.001

                # likelihood ratio test
                single_cluster_likelihood = 0
                two_clusters_likelihood = 0
                for i in range(num_samples_within_class):
                    single_cluster_likelihood += multivariate_normal.logpdf(x=projected_feats[i:i + 1], mean=mu[0],
                                                                            cov=covariance[0],
                                                                            allow_singular=True).sum()
                    two_clusters_likelihood += multivariate_normal.logpdf(x=projected_feats[i:i + 1], mean=mu[labels[i]],
                                                                        cov=covariance[labels[i]], allow_singular=True).sum()

                likelihood_ratio = np.exp( (two_clusters_likelihood - single_cluster_likelihood) / num_samples_within_class )

            else:

                likelihood_ratio = 1

            class_likelihood_ratio.append(likelihood_ratio)

            print('likelihood_ratio = ', likelihood_ratio)

        max_ratio = np.array(class_likelihood_ratio).max()

        for target_class in range(self.kettle.num_classes):
            likelihood_ratio = class_likelihood_ratio[target_class]

            if likelihood_ratio == max_ratio and likelihood_ratio > 1.5:  # a lower conservative threshold for maximum ratio

                print('[class-%d] class with maximal ratio %f!. Apply Cleanser!' % (target_class, max_ratio))

                for i in class_indices[target_class]:
                    if preds_inspection[i] == target_class:
                        suspicious_indices.append(i)

            elif likelihood_ratio > threshold:
                print('[class-%d] likelihood_ratio = %f > threshold = %f. Apply Cleanser!' % (
                    target_class, likelihood_ratio, threshold))

                for i in class_indices[target_class]:
                    if preds_inspection[i] == target_class:
                        suspicious_indices.append(i)

            else:
                print('[class-%d] likelihood_ratio = %f <= threshold = %f. Pass!' % (
                    target_class, likelihood_ratio, threshold))

        return suspicious_indices
        
def eval_model(model, kettle):
    model.eval()
    clean_acc, asr = 0, 0
    corrects = 0
    
    # Get device from model
    if hasattr(model, 'module'):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device
    
    for batch_idx, (data, target, idxs) in enumerate(tqdm(kettle.validloader)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        corrects += torch.eq(pred, target).sum().item()
    clean_acc = corrects / len(kettle.validloader.dataset)

    source_class = kettle.poison_setup['source_class'][0]
    target_class = kettle.poison_setup['target_class']

    corrects = 0
    for batch_idx, (data, _, _) in enumerate(tqdm(kettle.source_testloader[source_class])):
        data = data.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        corrects += torch.eq(pred, target_class).sum().item()
    asr = corrects / len(kettle.source_testloader[source_class].dataset)

    print(f"Clean Accuracy: {clean_acc*100:.2f}%, ASR: {asr*100:.2f}%")
    return clean_acc, asr