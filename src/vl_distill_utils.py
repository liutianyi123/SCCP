import os
import numpy as np
import torch
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import joblib
from PIL import Image

from diffusers import StableUnCLIPImg2ImgPipeline
from torch.utils.data import DataLoader, TensorDataset
from itertools import chain as ichain

from src.cluster_networks import Model_view, InstanceLoss, ClusterLoss

__all__ = [
    "nearest_neighbor",
    "load_or_process_file",
]


def nearest_neighbor(sentences, query_embeddings, database_embeddings):
    """
    Find the nearest neighbors for a batch of embeddings.
    """
    similarities = cosine_similarity(query_embeddings, database_embeddings)

    most_similar_indices = np.argmax(similarities, axis=1)

    nearest_neighbors = [sentences[i] for i in most_similar_indices]
        
    return nearest_neighbors



def load_or_process_file(file_type, process_func, args, data_source):
    """
    Load the processed file if it exists, otherwise process the data source and create the file.

    Args:
    file_type: The type of the file (e.g., 'train', 'test').
    process_func: The function to process the data source.
    args: The arguments required by the process function and to build the filename.
    data_source: The source data to be processed.

    Returns:
    The loaded data from the file.
    """
    if 'img' in file_type:
        filename = f'{args.embed_path}/{args.dataset}_{args.image_encoder}_{file_type}_embed.npz'
    elif 'text' in file_type:
        filename = f'{args.embed_path}/{args.dataset}_{args.text_encoder}_{file_type}_embed.npz'

    if not os.path.exists(filename):
        print(f'Creating {filename}')
        process_func(args, data_source)
    else:
        print(f'Loading {filename}')
    
    return np.load(filename)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_diversity_loss(probs):
    avg_probs = probs.mean(dim=0)
    entropy = - (avg_probs * torch.log(avg_probs + 1e-10)).sum()
    return -entropy


def clustering(img_embeds, txt_embeds, args):

    base_dir = f'data/center/{args.image_encoder}-{args.text_encoder}'
    work_dir = f'{base_dir}/{args.dataset}_{args.num_pairs}_pairs_{args.epochs}_epochs'

    os.makedirs(work_dir, exist_ok=True)

    img_center_path = os.path.join(work_dir, f'{args.dataset}_img_centers_{args.num_pairs}.pkl')
    txt_center_path = os.path.join(work_dir, f'{args.dataset}_text_centers_{args.num_pairs}.pkl')

    if not os.path.exists(img_center_path) or not os.path.exists(txt_center_path):
        print("Starting Training for feature alignment...")

        batch_size = args.batch_cluster
        lr_model = args.lr_model
        epochs = args.epochs
        layer_num = args.layer_num
        feature_dim = args.feature_dim
        num_pairs = args.num_pairs
        n_clusters = num_pairs

        dim_in = img_embeds.shape[1]

        dataset = TensorDataset(torch.tensor(img_embeds), torch.tensor(txt_embeds))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        models = {
            0: Model_view(n_clusters, dim_in, feature_dim).to(args.device),
            1: Model_view(n_clusters, dim_in, feature_dim).to(args.device)
        }

        ge_chain = ichain(*[list(models[i].parameters()) for i in range(2)])
        optimiser = torch.optim.Adam(ge_chain, lr=lr_model)

        instance_loss = InstanceLoss(batch_size, n_clusters, 0.1, args.device)
        cluster_loss = ClusterLoss(n_clusters, 0.1, args.device)

        for epoch in range(epochs):
            stats = {k: 0 for k in ['loss_total', 'loss_inst', 'loss_clus', 'loss_ent_div']}

            for batch, (data_img, data_txt) in enumerate(train_loader):
                data = {0: data_img.to(args.device), 1: data_txt.to(args.device)}
                features, clusters, h_distributions = {}, {}, {}

                for v in range(2):
                    features[v], clusters[v], h_distributions[v] = models[v](data[v])

                loss = 0
                loss_div = compute_diversity_loss(clusters[0]) + compute_diversity_loss(clusters[1])

                for i in range(2):
                    loss1, loss2 = 0, 0
                    j = 1 - i
                    for k in range(layer_num):
                        loss1 += instance_loss(features[i][k], features[j][k])
                    loss2 += cluster_loss(clusters[i], clusters[j])

                    loss += loss1 + loss2 + args.alpha * loss_div

                    stats['loss_inst'] += loss1.item()
                    stats['loss_clus'] += loss2.item()

                stats['loss_ent_div'] += loss_div.item()
                stats['loss_total'] += loss.item()

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            num_batches = len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Total: {stats['loss_total'] / num_batches:.3f} | "
                  f"Inst: {stats['loss_inst'] / num_batches:.3f} | "
                  f"Clus: {stats['loss_clus'] / num_batches:.3f} | "
                  f"Div: {stats['loss_ent_div'] / num_batches:.3f} ")

        print("Extracting features and calculating weighted centers...")
        for i in range(2): models[i].eval()

        prob_img_all, prob_txt_all = [], []
        hidden_img_all, hidden_txt_all = [], []

        with torch.no_grad():
            eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for data_img, data_txt in eval_loader:
                img_f, img_c, _ = models[0](data_img.to(args.device))
                txt_f, txt_c, _ = models[1](data_txt.to(args.device))

                prob_img_all.append(img_c.cpu().numpy())
                prob_txt_all.append(txt_c.cpu().numpy())
                hidden_img_all.append(img_f[layer_num - 1].cpu().numpy())
                hidden_txt_all.append(txt_f[layer_num - 1].cpu().numpy())

        prob_img = np.concatenate(prob_img_all, axis=0)
        prob_txt = np.concatenate(prob_txt_all, axis=0)
        hidden_img_embeds = np.concatenate(hidden_img_all, axis=0)
        hidden_txt_embeds = np.concatenate(hidden_txt_all, axis=0)

        img_labels = np.argmax(prob_img, axis=1)
        txt_labels = np.argmax(prob_txt, axis=1)

        img_centers = np.zeros((n_clusters, dim_in), dtype=np.float32)
        txt_centers = np.zeros((n_clusters, dim_in), dtype=np.float32)
        hidden_img_centers = np.zeros((n_clusters, feature_dim), dtype=np.float32)
        hidden_txt_centers = np.zeros((n_clusters, feature_dim), dtype=np.float32)

        for k in range(n_clusters):
            img_idx = np.where(img_labels == k)[0]
            txt_idx = np.where(txt_labels == k)[0]

            if len(img_idx) > 0:
                raw_w_img = prob_img[img_idx, k]
                sharpened_w_img = np.power(raw_w_img, 1.0 / args.tau)
                w_img = sharpened_w_img / (np.sum(sharpened_w_img) + 1e-8)
                img_centers[k] = np.sum(img_embeds[img_idx] * w_img[:, np.newaxis], axis=0)
                hidden_img_centers[k] = hidden_img_embeds[img_idx].mean(axis=0)

            if len(txt_idx) > 0:
                raw_w_txt = prob_txt[txt_idx, k]
                sharpened_w_txt = np.power(raw_w_txt, 1.0 / args.tau)
                w_txt = sharpened_w_txt / (np.sum(sharpened_w_txt) + 1e-8)
                txt_centers[k] = np.sum(txt_embeds[txt_idx] * w_txt[:, np.newaxis], axis=0)
                hidden_txt_centers[k] = hidden_txt_embeds[txt_idx].mean(axis=0)


        df = pd.DataFrame({'index': np.arange(len(img_labels)), 'img_cluster': img_labels, 'txt_cluster': txt_labels})
        count_table = df.groupby(['img_cluster', 'txt_cluster']).size().unstack(fill_value=0)
        cost_matrix = -count_table.values
        # Use for validating diagonal structure
        img_idxs, txt_idxs = linear_sum_assignment(cost_matrix)


        matched_img_centers, matched_txt_centers = match_and_sort_centers(
            img_labels, txt_labels, img_idxs, txt_idxs, img_centers, txt_centers, args
        )

        joblib.dump(matched_img_centers, img_center_path)
        joblib.dump(matched_txt_centers, txt_center_path)
        print("SDCIB weighted cluster centers saved.")

    return joblib.load(img_center_path), joblib.load(txt_center_path)


def match_and_sort_centers(img_labels, txt_labels, img_idxs, txt_idxs,
                           img_centers, txt_centers ,args):

    matched_img_centers = []
    matched_txt_centers = []

    empty_num = 0

    for i, j in zip(img_idxs, txt_idxs):
        mask = (img_labels == i) & (txt_labels == j)
        num_matched = np.sum(mask)
        print(f"Matched Image cluster {i} with Text cluster {j} -> {num_matched} samples")

        if num_matched == 0:
            empty_num = empty_num + 1

        matched_img_centers.append(img_centers[i])
        matched_txt_centers.append(txt_centers[j])

    print("Empty Cluster: ", empty_num)

    matched_img_centers = np.stack(matched_img_centers, axis=0)
    matched_txt_centers = np.stack(matched_txt_centers, axis=0)

    norm_img = matched_img_centers / (np.linalg.norm(matched_img_centers, axis=1, keepdims=True) + 1e-8)
    norm_txt = matched_txt_centers / (np.linalg.norm(matched_txt_centers, axis=1, keepdims=True) + 1e-8)

    sims_np = np.sum(norm_img * norm_txt, axis=1)
    sorted_indices = np.argsort(sims_np)[::-1]

    print(
        f"Centers sorted by similarity. Max sim: {sims_np[sorted_indices[0]]:.4f}, Min sim: {sims_np[sorted_indices[-1]]:.4f}")

    return matched_img_centers[sorted_indices], matched_txt_centers[sorted_indices]


def load_rep_embed(args, embed_type='text', get_origin=False):
    base_dir = f'data/center/{args.image_encoder}-{args.text_encoder}'
    work_dir = (f'{base_dir}/'
                f'{args.dataset}_{args.num_pairs}_pairs_{args.epochs}_epochs')

    img_center_path = os.path.join(work_dir, f'{args.dataset}_img_centers_{args.num_pairs}.pkl')
    txt_center_path = os.path.join(work_dir, f'{args.dataset}_text_centers_{args.num_pairs}.pkl')


    if embed_type == 'text':
        if not os.path.exists(txt_center_path):
            raise FileNotFoundError(f"Text centers file not found: {txt_center_path}")
        return joblib.load(txt_center_path)
    
    elif embed_type == 'image':
        if not os.path.exists(img_center_path):
            raise FileNotFoundError(f"Image centers file not found: {img_center_path}")
        return joblib.load(img_center_path)



def remove_low_sim_pairs(img_embeds, txt_embeds, sim, remove_ratio=0.1):
    assert len(img_embeds) == len(sim)
    assert 0 <= remove_ratio < 1

    num_to_remove = int(len(sim) * remove_ratio)
    if num_to_remove == 0:
        return img_embeds, txt_embeds

    sorted_indices = np.argsort(sim)
    remove_indices = sorted_indices[:num_to_remove]

    keep_mask = torch.ones(len(sim), dtype=torch.bool)
    keep_mask[remove_indices] = False

    return img_embeds[keep_mask], txt_embeds[keep_mask]


def compute_self_sim(img_embeds, txt_embeds, args, prune=False):
    assert len(img_embeds) == len(txt_embeds), "List lengths must match"

    norm_img = img_embeds / np.linalg.norm(img_embeds, axis=1, keepdims=True)
    norm_txt = txt_embeds / np.linalg.norm(txt_embeds, axis=1, keepdims=True)
    sims_np = np.sum(norm_img * norm_txt, axis=1) 
    
    return sims_np 



def generate_syn_img(img_emdeds, sentence_list, img_path, args):
    if sentence_list is not None:
        assert len(img_emdeds) == len(sentence_list), "Image and text embeddings must have the same length"

    decoder_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("sd2-community/stable-diffusion-2-1-unclip-small",
                                                               torch_dtype=torch.float16).to(args.device)

    if sentence_list is None:
        sentence_list = [""]*len(img_emdeds)  # Default empty prompt if none provided
    
    os.makedirs(f'{img_path}', exist_ok=True)
    for idx, (img_emded, txt_promt) in enumerate(zip(img_emdeds, sentence_list)):
        save_path = f'{img_path}/{idx}.png'
        img_emded = torch.tensor(img_emded, dtype=torch.float16).to(args.device)
        
        # Image generation using Unclip
        negative_prompt= "text, watermark"

        decoder_output = decoder_pipe(prompt=txt_promt, negative_prompt=negative_prompt, \
                                      image_embeds=img_emded.unsqueeze(0), num_inference_steps=args.infer_num_steps, \
                                      guidance_scale=args.guidance_scale, noise_level=args.noise_level)

        print(f"img_{idx} : {txt_promt}")
        
        img_generated = decoder_output.images[0]
            
        # Resize and save
        img_resized = img_generated.resize((args.image_size, args.image_size), resample=Image.LANCZOS)  #Image.NEAREST, Image.BILINEAR, Image.BICUBIC
        img_resized.save(save_path)
        
