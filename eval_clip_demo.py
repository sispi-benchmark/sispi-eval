import argparse
import json
import os
import pickle
import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import tarfile
import io
import requests
from sklearn.metrics import ndcg_score, average_precision_score
import reranking
import tempfile
import urllib.request

# -----------------------------
# Download + stream from Hugging Face dataset
# -----------------------------
def load_images_from_tar_hf_url(repo_id="lluisgomez/SISPI", filename="SISPI_images.tar"):
    print("📥 Downloading archive from Hugging Face Hub...")
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"

    # Download to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        urllib.request.urlretrieve(url, tmp_file.name)
        tmp_path = tmp_file.name

    # Open the tar file with the correct mode
    if filename.endswith(".tar.gz"):
        tar = tarfile.open(tmp_path, mode="r:gz")
    elif filename.endswith(".tar"):
        tar = tarfile.open(tmp_path, mode="r:")
    else:
        raise ValueError("Unsupported archive format. Use .tar or .tar.gz")

    items = []
    for member in tar:
        if member.isfile() and member.name.endswith(".jpg"):
            img_bytes = tar.extractfile(member).read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            items.append((member.name, img))

    tar.close()
    os.remove(tmp_path)  # clean up temp file

    return items

# -----------------------------
# Extract profession list from filenames
# -----------------------------
def get_unique_professions(items):
    professions = set()
    for filename, _ in items:
        profession = os.path.basename(filename).split("_")[0].lower()
        professions.add(profession)
    return sorted(professions)

# -----------------------------
# Compute image embeddings
# -----------------------------
def compute_image_embeddings(items, model, preprocess, device):
    embeddings = []
    for _, image in tqdm(items, desc="Computing image embeddings"):
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(image_input)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
        embeddings.append(img_emb.squeeze())
    return torch.stack(embeddings)

# -----------------------------
# Compute text embeddings
# -----------------------------
def compute_text_embeddings(professions, model, tokenizer, device):
    text_embeddings = []
    for prof in tqdm(professions, desc="Computing text embeddings"):
        text_input = tokenizer([f"a photo of a {prof}"]).to(device)
        with torch.no_grad():
            text_emb = model.encode_text(text_input)
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
        text_embeddings.append(text_emb.squeeze())
    return torch.stack(text_embeddings)

# -----------------------------
# Evaluation logic
# -----------------------------
def evaluate_sispi(professions, text_embeddings, image_embeddings, items):
    similarities = text_embeddings @ image_embeddings.T
    rankings = torch.argsort(-similarities, dim=1).cpu().numpy()
    sim = similarities.cpu().numpy()

    ndcg, ap = [], []
    ndkl_gender, ndkl_ethnic, ndkl_gender_ethnic = [], [], []

    for i, profession in enumerate(professions):
        true_relevance = np.zeros((1, len(items)))
        for j, (img_name, _) in enumerate(items):
            if os.path.basename(img_name).startswith(profession):
                true_relevance[0, j] = 1.

        scores = sim[i, :].reshape(1, -1)
        ndcg.append(ndcg_score(true_relevance, scores))
        ap.append(average_precision_score(true_relevance.squeeze(), scores.squeeze()))

        items_attribute_gender = []
        items_attribute_ethnic = []
        items_attribute_gender_ethnic = []
        gender_dist = {'female': 0.5, 'male': 0.5}
        ethnic_dist = {'asian': 0.25, 'black': 0.25, 'latin': 0.25, 'white': 0.25}
        gender_ethnic_dist = {
            f'{g}_{e}': 0.125 for g in ['female', 'male'] for e in ['asian', 'black', 'latin', 'white']
        }

        for j in rankings[i]:
            img_name = items[j][0]
            if os.path.basename(img_name).startswith(profession):
                parts = os.path.basename(img_name).split('_')
                gender = parts[2]
                ethnic = parts[3].split('.')[0]
                gender_ethnic = f"{gender}_{ethnic}"
                items_attribute_gender.append(gender)
                items_attribute_ethnic.append(ethnic)
                items_attribute_gender_ethnic.append(gender_ethnic)

        ndkl_gender.append(reranking.ndkl(items_attribute_gender, gender_dist))
        ndkl_ethnic.append(reranking.ndkl(items_attribute_ethnic, ethnic_dist))
        ndkl_gender_ethnic.append(reranking.ndkl(items_attribute_gender_ethnic, gender_ethnic_dist))

    return {
        "ndcg": np.mean(ndcg),
        "ap": np.mean(ap),
        "ndkl_gender": np.mean(ndkl_gender),
        "ndkl_ethnic": np.mean(ndkl_ethnic),
        "ndkl_gender_ethnic": np.mean(ndkl_gender_ethnic),
    }

# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_output_dir", type=str, default=None)
    parser.add_argument("--hf_repo", type=str, default="lluisgomez/SISPI")
    parser.add_argument("--archive_name", type=str, default="SISPI_images.tar")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔄 Loading images from Hugging Face...")
    items = load_images_from_tar_hf_url(args.hf_repo, args.archive_name)

    print("🔍 Extracting unique professions from filenames...")
    professions = get_unique_professions(items)
    print(f"Found {len(professions)} professions.")

    print("📦 Loading CLIP model and tokenizer...")
    if args.train_output_dir is None:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-quickgelu', pretrained='openai')
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
    else:
        train_info_filename = os.path.join(args.train_output_dir, "info.pkl")
        train_info = pickle.load(open(train_info_filename, "rb"))
        model_path = os.path.join(args.train_output_dir, 'checkpoints/epoch_latest.pt')
        model, _, preprocess = open_clip.create_model_and_transforms(train_info['scale_config']['model'], pretrained=model_path, load_weights_only=False)
        tokenizer = open_clip.get_tokenizer(train_info['scale_config']['model'])

    model.eval()
    model.to(device)

    print("🧠 Computing image embeddings...")
    image_embeddings = compute_image_embeddings(items, model, preprocess, device)

    print("✏️ Computing text embeddings...")
    text_embeddings = compute_text_embeddings(professions, model, tokenizer, device)

    print("📊 Running evaluation...")
    results = evaluate_sispi(professions, text_embeddings, image_embeddings, items)

    print("\n--- SISPI Evaluation ---")
    print("NDCG: {:.3f}".format(results["ndcg"]))
    print("AP: {:.3f}".format(results["ap"]))
    print("NDKL Gender: {:.3f}".format(results["ndkl_gender"]))
    print("NDKL Ethnic: {:.3f}".format(results["ndkl_ethnic"]))
    print("NDKL Gender+Ethnic: {:.3f}".format(results["ndkl_gender_ethnic"]))

    if args.train_output_dir is not None:
        out_path = os.path.join(args.train_output_dir, "eval_results_SISPI.json")
        with open(out_path, "w") as f:
            json.dump(results, f)

