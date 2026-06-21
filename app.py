import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# --- Import TensorFlow/Keras (Wajib) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from PIL import Image
except ImportError as e:
    raise SystemExit(
        f"[ERROR FATAL] Library ML tidak tersedia: {e}\n"
        "Pastikan TensorFlow dan Pillow sudah terinstall:\n"
        "  pip install tensorflow pillow"
    )

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimal ukuran unggahan 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Pastikan folder unggahan ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load class_names.json ---
CLASS_INFO = {}
CLASS_LIST = []
class_names_path = os.path.join('model', 'class_names.json')

if os.path.exists(class_names_path):
    with open(class_names_path, 'r', encoding='utf-8') as f:
        CLASS_INFO = json.load(f)
        CLASS_LIST = list(CLASS_INFO.keys())
    print(f"[OK] Berhasil memuat {len(CLASS_LIST)} kelas dari class_names.json: {CLASS_LIST}")
else:
    raise SystemExit(
        f"[ERROR FATAL] File class_names.json tidak ditemukan di: {class_names_path}\n"
        "Pastikan file tersebut ada di folder model/"
    )

# --- Load Model Keras ---
MODEL_PATH = os.path.join('model', 'model_penyakit_kulit.keras')
model = None

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        f"[ERROR FATAL] File model tidak ditemukan di: {MODEL_PATH}\n"
        "Pastikan file model_penyakit_kulit.keras ada di folder model/"
    )

try:
    model = load_model(MODEL_PATH)
    # Baca input size dari model secara otomatis
    MODEL_INPUT_SHAPE = model.input_shape  # e.g. (None, 224, 224, 3)
    IMG_HEIGHT = MODEL_INPUT_SHAPE[1]
    IMG_WIDTH  = MODEL_INPUT_SHAPE[2]
    print(f"[OK] Berhasil memuat model: {MODEL_PATH}")
    print(f"[OK] Input size model: {IMG_WIDTH}x{IMG_HEIGHT}")
except Exception as e:
    raise SystemExit(
        f"[ERROR FATAL] Gagal memuat model Keras: {e}\n"
        "Periksa apakah file .keras tidak rusak atau tidak kompatibel."
    )

# --- Load MobileNetV2 (untuk validasi gambar) ---
mobile_model = None
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # pyrefly: ignore [missing-import]
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    mobile_model = MobileNetV2(weights='imagenet')
    print("[OK] Berhasil memuat MobileNetV2 untuk validasi gambar.")
except Exception as e:
    print(f"[WARNING] Gagal memuat MobileNetV2: {e}. Validasi gambar dinonaktifkan.")

# ImageNet indeks untuk hewan (mamalia, burung, reptil yang mungkin ada dalam foto kulit hewan)
# Range 0-397 mencakup sebagian besar kelas hewan di ImageNet
# 151-268: anjing berbagai ras, 281-293: kucing, 80-100: burung, dll.
# Tambahkan indeks lain agar filter lebih luas dan tidak over-reject foto kulit close-up
ANIMAL_INDICES = set(range(0, 400))   # 0-399 = hampir semua kelas hewan di ImageNet
# Kelas NON-hewan yang ingin kita tolak: manusia, benda, bangunan, makanan, kendaraan, dll.
# ImageNet kelas manusia: 0 (treadmill), sekitar 400-999 mayoritas benda/manusia/bangunan
# Kita tolak gambar jika top prediction jelas bukan hewan (kelas 500+) DAN prob hewan sangat rendah
OBVIOUS_NON_ANIMAL = set(range(500, 1000))  # benda mati, manusia, bangunan, kendaraan, makanan

def is_relevant_image(filepath):
    """
    Validasi minimal: tolak hanya gambar yang JELAS bukan foto hewan sama sekali.
    (screenshot, foto manusia, bangunan, makanan, kendaraan, pemandangan tanpa hewan)
    Gambar kulit hewan close-up HARUS diterima meski MobileNetV2 kurang yakin.
    Returns dict: {'valid': bool, 'message': str}
    """
    if mobile_model is None:
        return {'valid': True, 'message': ''}
    try:
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))
        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = mobile_model.predict(x, verbose=0)[0]
        top_indices = np.argsort(preds)[::-1][:5]  # 5 kelas teratas
        top_idx    = int(top_indices[0])
        top_prob   = float(preds[top_idx])

        # Probabilitas gabungan untuk kelas hewan (0-399)
        animal_prob = float(sum(preds[i] for i in top_indices if i < 400))

        print(f"[DEBUG] top_idx={top_idx}, top_prob={top_prob:.3f}, animal_prob={animal_prob:.3f}")

        # Tolak HANYA jika kelas teratas adalah non-hewan (>=500) DAN
        # total prob hewan sangat rendah (<5%) — ini berarti foto jelas bukan hewan
        if top_idx >= 500 and animal_prob < 0.05:
            return {
                'valid': False,
                'message': 'Gambar tidak relevan. Unggah foto kulit kucing atau anjing yang jelas.'
            }

        return {'valid': True, 'message': ''}

    except Exception as e:
        print(f"[ERROR] Validasi gambar gagal: {e}")
        return {'valid': True, 'message': ''}  # Fallback — jangan blokir jika error

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', is_mock=False, class_info=CLASS_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file gambar yang dikirim'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nama file kosong'}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({'success': False, 'error': 'Format file tidak diizinkan. Gunakan JPG, JPEG, atau PNG'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    pet_type = request.form.get('pet_type', 'kucing').lower()

    # --- Validasi Gambar: Tolak hanya jika jelas bukan foto hewan ---
    validation = is_relevant_image(filepath)
    if not validation['valid']:
        # Hapus file yang tidak valid untuk menghemat ruang
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as err:
            print(f"[WARNING] Gagal menghapus file tidak valid: {err}")
        return jsonify({
            'success': False,
            'status': 'invalid',
            'error': validation['message'],
            'detected_species': validation.get('detected_species')
        }), 400

    try:
        # --- Preprocessing Gambar ---
        img = Image.open(filepath).convert('RGB')
        img = img.resize((224, 224))
        x = np.array(img, dtype=np.float32)
        x = np.expand_dims(x, axis=0)


        # --- Prediksi ---
        preds = model.predict(x, verbose=0)[0]  # Array probabilitas tiap kelas
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])

        # Petakan index ke nama kelas
        class_key = CLASS_LIST[pred_idx] if pred_idx < len(CLASS_LIST) else CLASS_LIST[-1]

        # Buat daftar semua probabilitas untuk chart
        all_predictions = []
        for idx, c_key in enumerate(CLASS_LIST):
            prob = float(preds[idx]) if idx < len(preds) else 0.0
            info = CLASS_INFO.get(c_key, {"name": c_key})
            all_predictions.append({
                'class_key': c_key,
                'name': info.get('name', c_key),
                'confidence': round(prob * 100, 2)
            })
        all_predictions = sorted(all_predictions, key=lambda p: p['confidence'], reverse=True)

        # Hitung selisih 2 kelas tertinggi
        top_1 = all_predictions[0]
        top_2 = all_predictions[1] if len(all_predictions) > 1 else {"confidence": 0.0, "name": "", "class_key": ""}
        
        top_confidence = top_1['confidence']
        sec_confidence = top_2['confidence']
        diff_confidence = top_confidence - sec_confidence

        # Evaluasi flags
        is_low_confidence = top_confidence < 75.0
        is_ambiguous = diff_confidence < 15.0

        warning_messages = []
        if is_low_confidence:
            warning_messages.append("Model belum yakin dengan hasil prediksi. Unggah foto kulit yang lebih dekat, fokus, dan pencahayaan baik.")
        if is_ambiguous:
            warning_messages.append("Kemungkinan terdapat lebih dari satu kondisi kulit. Hasil ini belum dapat dijadikan diagnosis pasti.")

        # Debug Logs di Backend console
        print(f"[DEBUG] predicted_class: {top_1['class_key']}", flush=True)
        print(f"[DEBUG] confidence: {top_confidence}%", flush=True)
        print("[DEBUG] seluruh probabilitas kelas:", flush=True)
        for pred in all_predictions:
            print(f"  - {pred['class_key']} ({pred['name']}): {pred['confidence']}%", flush=True)

    except Exception as e:
        print(f"[ERROR] Prediksi gagal: {e}")
        return jsonify({
            'success': False,
            'status': 'invalid',
            'error': f'Prediksi gagal: {str(e)}. Pastikan gambar valid dan coba lagi.'
        }), 500

    disease_details = CLASS_INFO.get(class_key, {
        "name": class_key,
        "description": "Detail informasi penyakit kulit ini belum tersedia.",
        "symptoms": ["Iritasi atau kelainan pada kulit."],
        "treatment": ["Konsultasikan ke dokter hewan terdekat."],
        "danger_level": "Tidak diketahui"
    })

    return jsonify({
        'success': True,
        'status': 'success',
        'is_mock': False,
        'prediction': {
            'class_key': class_key,
            'name': disease_details['name'],
            'confidence': round(confidence * 100, 2),
            'description': disease_details['description'],
            'symptoms': disease_details['symptoms'],
            'treatment': disease_details['treatment'],
            'danger_level': disease_details['danger_level']
        },
        'all_predictions': all_predictions,
        'image_url': f'/static/uploads/{filename}',
        'pet_type': pet_type,
        'is_low_confidence': is_low_confidence,
        'is_ambiguous': is_ambiguous,
        'warning_messages': warning_messages
    })

# Mock mode telah dihapus. Aplikasi selalu menggunakan model TensorFlow asli.

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)
