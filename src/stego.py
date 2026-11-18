import os
import json
import csv
import struct
import hashlib
import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.KDF import PBKDF2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import chi2 as chi2_dist
import matplotlib.pyplot as plt


# ==========================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (LSB)
# ==========================================
def bytes_to_bits(data: bytes):
    return [b for byte in data for b in format(byte, '08b')]


def bits_to_bytes(bits):
    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte_array.append(int(''.join(bits[i:i + 8]), 2))
    return bytes(byte_array)


def get_rng(password):
    """Детерминированный генератор для псевдослучайных позиций."""
    seed = int(hashlib.sha256(password.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def embed_lsb(img_array, payload_bytes, positions=None):
    flat_img = img_array.flatten()
    data_with_len = struct.pack('>I', len(payload_bytes)) + payload_bytes
    bits = bytes_to_bits(data_with_len)

    if len(bits) > len(flat_img):
        raise ValueError("Сообщение слишком велико для данного изображения!")

    if positions is None:
        positions = range(len(bits))

    for i, bit in enumerate(bits):
        pos = positions[i]
        flat_img[pos] = (flat_img[pos] & 254) | int(bit)

    return flat_img.reshape(img_array.shape)


def extract_lsb(img_array, password=None):
    """Извлекает данные. Если password=None, то последовательно. Иначе - псевдослучайно."""
    flat_img = img_array.flatten()

    if password is None:
        positions = range(len(flat_img))
    else:
        # Восстанавливаем ту же последовательность, что и при встраивании
        rng = get_rng(password)
        positions = rng.permutation(len(flat_img))

    # Читаем длину (первые 32 бита)
    len_bits = [str(flat_img[positions[i]] & 1) for i in range(32)]
    payload_len = struct.unpack('>I', bits_to_bytes(len_bits))[0]

    # Читаем сами данные
    data_bits = []
    for i in range(32, 32 + payload_len * 8):
        data_bits.append(str(flat_img[positions[i]] & 1))

    return bits_to_bytes(data_bits)


# ==========================================
# 2. ШИФРОВАНИЕ
# ==========================================
def encrypt_payload(text, password):
    salt = os.urandom(16)
    key = PBKDF2(password, salt, dkLen=32)
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ct = cipher.encrypt(pad(text.encode('utf-8'), AES.block_size))
    return salt + iv + ct


def decrypt_payload(encrypted_data, password):
    salt = encrypted_data[:16]
    iv = encrypted_data[16:32]
    ct = encrypted_data[32:]
    key = PBKDF2(password, salt, dkLen=32)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ct), AES.block_size)
    return pt.decode('utf-8')


# ==========================================
# 3. МЕТРИКИ
# ==========================================
def calc_lsb_distribution(img_array):
    flat = img_array.flatten()
    lsb_ones = np.sum(flat & 1)
    total = len(flat)
    return (total - lsb_ones) / total, lsb_ones / total


def calc_chi2(img_array):
    """
    H0: Распределение пар (2k, 2k+1) равномерное.
    Формула: chi2 = sum((Obs - Exp)^2 / Exp)
    df = 127 (так как пар 128).
    """
    hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
    expected = (hist[0::2] + hist[1::2]) / 2.0
    valid = expected > 0
    chi2_stat = np.sum(((hist[0::2][valid] - expected[valid]) ** 2) / expected[valid])
    df = np.sum(valid) - 1
    p_value = 1 - chi2_dist.cdf(chi2_stat, df=df)
    return chi2_stat, p_value


def calc_changed_ratio(cover, stego):
    changed = np.sum(cover != stego)
    # Возвращаем в процентах
    return changed, (changed / cover.size) * 100


# ==========================================
# 4. ГЛАВНАЯ ЧАСТЬ
# ==========================================
if __name__ == "__main__":
    print("ЗАПУСК СКРИПТА...")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    imgs_dir = os.path.join(base_dir, "imgs")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    images = ["my_photo.png", "gradient.png", "checkerboard.png"]
    message = "Секретное сообщение для лабораторной работы! Это очень важный текст."
    password = "SuperSecretPassword123"

    all_metrics = []

    for img_name in images:
        cover_path = os.path.join(imgs_dir, img_name)
        if not os.path.exists(cover_path):
            print(f"Пропуск: {cover_path} не найден")
            continue

        print(f"\n=== Обработка {img_name} ===")
        cover_img = Image.open(cover_path).convert('RGB')
        cover_arr = np.array(cover_img)
        stem = os.path.splitext(img_name)[0]

        # --- 1. Plain LSB ---
        stego_plain_arr = embed_lsb(cover_arr, message.encode('utf-8'))
        Image.fromarray(stego_plain_arr).save(os.path.join(imgs_dir, f"{stem}_stego_plain.png"))
        assert extract_lsb(stego_plain_arr).decode('utf-8') == message

        # --- 2. Encrypted LSB ---
        encrypted_msg = encrypt_payload(message, password)
        stego_enc_arr = embed_lsb(cover_arr, encrypted_msg)
        Image.fromarray(stego_enc_arr).save(os.path.join(imgs_dir, f"{stem}_stego_encrypted.png"))
        assert decrypt_payload(extract_lsb(stego_enc_arr), password) == message

        # --- 3. Encrypted Random LSB ---
        rng = get_rng(password)
        needed_bits = len(encrypted_msg) * 8 + 32
        random_positions = rng.permutation(cover_arr.size)[:needed_bits]

        stego_rand_arr = embed_lsb(cover_arr, encrypted_msg, random_positions)
        Image.fromarray(stego_rand_arr).save(os.path.join(imgs_dir, f"{stem}_stego_random.png"))

        # ВАЖНО: Извлекаем только по паролю, без передачи random_positions!
        extracted_rand = extract_lsb(stego_rand_arr, password=password)
        assert decrypt_payload(extracted_rand, password) == message
        print("   Все тесты на извлечение (включая Random LSB) пройдены.")

        # --- МЕТРИКИ ---
        variants = {
            "Plain LSB": stego_plain_arr,
            "Encrypted LSB": stego_enc_arr,
            "Random LSB": stego_rand_arr
        }

        for name, arr in variants.items():
            p = round(psnr(cover_arr, arr, data_range=255), 2)
            s = round(ssim(cover_arr, arr, channel_axis=2, data_range=255), 4)
            ch, ch_r = calc_changed_ratio(cover_arr, arr)
            p0, p1 = calc_lsb_distribution(arr)
            chi2_stat, chi2_p = calc_chi2(arr)

            all_metrics.append({
                "Image": img_name,
                "Variant": name,
                "Payload Rate": "Small",
                "PSNR": p,
                "SSIM": s,
                "Changed Ratio (%)": round(ch_r, 2),
                "P(LSB=1)": round(p1, 4),
                "Chi2": round(chi2_stat, 2),
                "p-value": round(chi2_p, 4)
            })

        # --- ВЛИЯНИЕ ОБЪЕМА ДАННЫХ (1% и 5%) ---
        capacity_bits = cover_arr.size
        for fill_rate in [0.01, 0.05]:
            target_bits = int(capacity_bits * fill_rate)
            text_len = (target_bits - 32) // 8
            large_text = "A" * text_len

            stego_large = embed_lsb(cover_arr, large_text.encode('utf-8'))
            p_large = round(psnr(cover_arr, stego_large, data_range=255), 2)
            _, ch_r_large = calc_changed_ratio(cover_arr, stego_large)
            _, p_val_large = calc_chi2(stego_large)

            all_metrics.append({
                "Image": img_name,
                "Variant": f"Plain LSB ({int(fill_rate * 100)}%)",
                "Payload Rate": f"{int(fill_rate * 100)}%",
                "PSNR": p_large,
                "SSIM": "-",
                "Changed Ratio (%)": round(ch_r_large, 2),
                "P(LSB=1)": "-",
                "Chi2": "-",
                "p-value": round(p_val_large, 4)
            })
            print(f"   Заполнение {int(fill_rate * 100)}%: PSNR={p_large} дБ, p-value={p_val_large:.4f}")

        # --- ВИЗУАЛИЗАЦИЯ (только для my_photo) ---
        if img_name == "my_photo.png":
            diff = np.abs(cover_arr.astype(int) - stego_plain_arr.astype(int)).astype(np.uint8)
            Image.fromarray(np.clip(diff * 50, 0, 255).astype(np.uint8)).save(
                os.path.join(results_dir, "difference.png"))

            plt.figure(figsize=(10, 6))
            sample_cover = np.random.choice(cover_arr.flatten(), size=50000, replace=False)
            sample_stego = np.random.choice(stego_rand_arr.flatten(), size=50000, replace=False)
            plt.hist(sample_cover, bins=256, alpha=0.5, label='Cover', color='blue')
            plt.hist(sample_stego, bins=256, alpha=0.5, label='Random LSB', color='red')
            plt.title("Cover vs Random LSB")
            plt.legend()
            plt.savefig(os.path.join(results_dir, "histogram.png"))
            plt.close()

    # --- СОХРАНЕНИЕ СВОДНОГО CSV ---
    metrics_path = os.path.join(results_dir, "metrics.csv")
    with open(metrics_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nГОТОВО! Все данные сохранены в {metrics_path}")