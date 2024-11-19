import os
import cv2
import numpy as np
import dlib
import pickle
#Tiến hành chụp ảnh, biến đổi ảnh.Sau đó tiến hành trích xuất đặc chưng của đối tượng trong ảnh ra file .pkl(pickle)
# Khởi tạo camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở camera.")
    exit()

# Yêu cầu người dùng nhập tên đối tượng
object_name = input("Nhập tên đối tượng mới: ").strip()

# Đặt đường dẫn lưu ảnh
save_directory = f'./data/imagefaceid/{object_name}'
original_directory = os.path.join(save_directory, "original_images")
transformed_directory = os.path.join(save_directory, "transformed_images")
os.makedirs(original_directory, exist_ok=True)
os.makedirs(transformed_directory, exist_ok=True)

# Các góc chụp
angle_positions = ['NhinThang']
current_position = 0

# Bắt đầu hiển thị video từ camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc khung hình.")
        break

    # Lật khung hình theo chiều ngang
    flipped_frame = cv2.flip(frame, 1)

    # Hiển thị hướng dẫn góc quay mặt
    cv2.putText(flipped_frame, f'Huong dan: {angle_positions[current_position]}', 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Hiển thị khung hình
    cv2.imshow('Nguồn Camera', flipped_frame)

    key = cv2.waitKey(1)
    
    # Chụp ảnh khi người dùng nhấn phím 'Space'
    if key == 32:  # Phím Space
        photo_filename = os.path.join(original_directory, f'{object_name}_{angle_positions[current_position]}.png')
        cv2.imwrite(photo_filename, flipped_frame)
        print(f"Đã chụp ảnh và lưu vào '{photo_filename}'.")

        # Chuyển sang góc tiếp theo
        current_position += 1
        if current_position >= len(angle_positions):
            print("Đã hoàn tất chụp ảnh ở tất cả các góc.") 
            break

    # Thoát vòng lặp nếu người dùng nhấn phím 'Esc'
    if key == 27:  # Phím Esc
        break

# Định nghĩa các hàm biến đổi ảnh
def flip_image(image):
    return cv2.flip(image, 1)

def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def negative_image(image):
    return cv2.bitwise_not(image)

def log_transform(image):
    epsilon = 1e-5
    c = 255 / np.log(1 + np.max(image))
    log_image = c * np.log(image + 1 + epsilon)
    return np.array(log_image, dtype=np.uint8)

def histogram_equalization(image):
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)
    else:
        channels = cv2.split(image)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]
        return cv2.merge(eq_channels)

def resize_image(image, scale_x, scale_y):
    return cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Thực hiện biến đổi trên mỗi ảnh đã chụp và lưu lại vào thư mục transformed_images
for image_file in os.listdir(original_directory):
    img_path = os.path.join(original_directory, image_file)
    
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        transformations = {
            "flip": flip_image(img),
            "rotate_90": rotate_image(img, 90),
            "rotate_180": rotate_image(img, 180),
            "rotate_270": rotate_image(img, 270),
            "negative": negative_image(img),
            "log_transform": log_transform(img),
            "histogram_equalization": histogram_equalization(img),
            "stretch": resize_image(img, 1.5, 1.5),
            "shrink": resize_image(img, 0.5, 0.5)
        }
        
        for transform_name, transformed_img in transformations.items():
            output_path = os.path.join(transformed_directory, f"{os.path.splitext(image_file)[0]}_{transform_name}.png")
            cv2.imwrite(output_path, transformed_img)

print("Hoàn tất xử lý: ảnh gốc đã lưu trong 'original_images' và ảnh biến đổi trong 'transformed_images'.")

# Bước 3: Trích xuất đặc điểm thông tin đối tượng đã chụp
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./dat/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("./dat/dlib_face_recognition_resnet_model_v1.dat")

image_folder = transformed_directory
known_face_descriptors = []

# Trích xuất đặc trưng và lưu tên và đặc trưng vào danh sách
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    img = cv2.imread(image_path)

    if img is None:
        continue

    faces = detector(img)
    for face in faces:
        shape = shape_predictor(img, face)
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
        known_face_descriptors.append((object_name, face_descriptor))  # Sử dụng object_name

# Lưu dữ liệu vào tệp pickle
with open("face_descriptors.pkl", "wb") as f:
    pickle.dump(known_face_descriptors, f)

print("Lưu đặc trưng khuôn mặt vào tệp thành công!")

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
