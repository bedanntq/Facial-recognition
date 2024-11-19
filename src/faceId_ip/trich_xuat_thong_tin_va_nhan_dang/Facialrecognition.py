# import dlib
# import cv2
# import numpy as np
# import pickle
# import threading
# import queue
# import time

# # Khởi tạo bộ phát hiện khuôn mặt và các mô hình
# detector = dlib.get_frontal_face_detector()
# shape_predictor = dlib.shape_predictor("./dat/shape_predictor_68_face_landmarks.dat")
# face_rec_model = dlib.face_recognition_model_v1("./dat/dlib_face_recognition_resnet_model_v1.dat")

# # Tải đặc trưng từ tệp đã lưu
# with open("face_descriptors.pkl", "rb") as f:
#     known_face_descriptors = pickle.load(f)

# # Hàm tính khoảng cách giữa hai mã đặc trưng
# def euclidean_distance(face_descriptor1, face_descriptor2):
#     return np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))

# # Khởi tạo camera và hàng đợi (queue) để truyền khung hình
# video_capture = cv2.VideoCapture(0)
# frame_queue = queue.Queue(maxsize=5)  # Giới hạn hàng đợi

# # Thiết lập độ phân giải cho khung hình camera
# frame_width = 240
# frame_height = 180
# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# # Biến cờ để dừng luồng
# stop_threads = threading.Event()

# # Luồng đọc khung hình từ camera
# def read_frames():
#     while not stop_threads.is_set():
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         # Đặt khung hình vào hàng đợi, nếu đầy thì bỏ khung hình cũ
#         if not frame_queue.full():
#             frame_queue.put(frame)
#         else:
#             frame_queue.get()  # Bỏ khung hình cũ để giữ hàng đợi không bị đầy
#             frame_queue.put(frame)

# # Luồng xử lý khung hình
# def process_frames():
#     while not stop_threads.is_set():
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             faces = detector(frame)
            
#             for face in faces:
#                 shape = shape_predictor(frame, face)
#                 face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
                
#                 # So sánh với đặc trưng đã lưu
#                 name = "Unknown"
#                 min_distance = 0.6  # Ngưỡng khoảng cách
#                 for known_name, known_descriptor in known_face_descriptors:
#                     distance = euclidean_distance(face_descriptor, known_descriptor)
#                     if distance < min_distance:
#                         min_distance = distance
#                         name = known_name
                
#                 # Vẽ khung và tên lên màn hình
#                 (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Hiển thị khung hình đã xử lý
#             cv2.imshow("Face Recognition", frame)

#             # Thoát khi nhấn 'q'
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 stop_threads.set()  # Đặt cờ để dừng các luồng
#                 break

# # Tạo và chạy các luồng
# read_thread = threading.Thread(target=read_frames)
# process_thread = threading.Thread(target=process_frames)
# read_thread.start()
# process_thread.start()

# # Chờ luồng kết thúc
# read_thread.join()
# process_thread.join()

# # Giải phóng tài nguyên
# video_capture.release()
# cv2.destroyAllWindows()

# ===============================================================================================

# import dlib
# import cv2
# import numpy as np
# import pickle
# import threading
# import queue
# import firebase_admin
# from firebase_admin import credentials, db

# # Firebase initialization
# cred = credentials.Certificate("./json/serviesAcc.json")
# firebase_admin.initialize_app(cred, {
#     "databaseURL": "https://aidatalist-default-rtdb.asia-southeast1.firebasedatabase.app/"
# })
# ref = db.reference("/users")  # Truy cập vào nút 'users'

# # Face recognition initialization
# detector = dlib.get_frontal_face_detector()
# shape_predictor = dlib.shape_predictor("./dat/shape_predictor_68_face_landmarks.dat")
# face_rec_model = dlib.face_recognition_model_v1("./dat/dlib_face_recognition_resnet_model_v1.dat")

# # Load known face descriptors
# with open("face_descriptors.pkl", "rb") as f:
#     known_face_descriptors = pickle.load(f)

# def euclidean_distance(face_descriptor1, face_descriptor2):
#     return np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))

# # Initialize camera and queue
# video_capture = cv2.VideoCapture(0)
# frame_queue = queue.Queue(maxsize=5)

# # Camera resolution
# frame_width = 240
# frame_height = 180
# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# # Stop threads flag
# stop_threads = threading.Event()

# # Biến lưu trữ tên đối tượng đã xuất gần nhất
# last_recognized_name = None

# # Thread for reading frames
# def read_frames():
#     while not stop_threads.is_set():
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         if not frame_queue.full():
#             frame_queue.put(frame)
#         else:
#             frame_queue.get()
#             frame_queue.put(frame)

# # Thread for processing frames
# def process_frames():
#     global last_recognized_name
#     while not stop_threads.is_set():
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             faces = detector(frame)
            
#             for face in faces:
#                 shape = shape_predictor(frame, face)
#                 face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
                
#                 # Face recognition
#                 name = "Unknown"
#                 min_distance = 0.6
#                 for known_name, known_descriptor in known_face_descriptors:
#                     distance = euclidean_distance(face_descriptor, known_descriptor)
#                     if distance < min_distance:
#                         min_distance = distance
#                         name = known_name
                
#                 # Fetch data from Firebase and save to file if recognized
#                 if name != "Unknown" and name != last_recognized_name:
#                     last_recognized_name = name
#                     data = ref.child(name).get()
#                     if data:
#                         output_file = f"{name}_info.txt"
#                         with open(output_file, "w", encoding="utf-8") as file:
#                             file.write(str({name: data}))
#                         print(f"Thông tin của '{name}' đã được lưu vào file {output_file}.")
#                     else:
#                         print(f"Không tìm thấy thông tin của '{name}' trong Firebase.")
                
#                 # Draw rectangle and name on the frame
#                 (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Display the frame
#             cv2.imshow("Face Recognition", frame)

#             # Exit on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 stop_threads.set()
#                 break

# # Start threads
# read_thread = threading.Thread(target=read_frames)
# process_thread = threading.Thread(target=process_frames)
# read_thread.start()
# process_thread.start()

# # Wait for threads to finish
# read_thread.join()
# process_thread.join()

# # Release resources
# video_capture.release()
# cv2.destroyAllWindows()
# ===============================================================================================
import dlib
import cv2
import numpy as np
import pickle
import threading
import queue
import firebase_admin
from firebase_admin import credentials, db

# Firebase initialization
cred = credentials.Certificate("./json/serviesAcc.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://aidatalist-default-rtdb.asia-southeast1.firebasedatabase.app/"
})
ref = db.reference("/users")  # Truy cập vào nút 'users'

# Face recognition initialization
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("./dat/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("./dat/dlib_face_recognition_resnet_model_v1.dat")

# Load known face descriptors
with open("face_descriptors.pkl", "rb") as f:
    known_face_descriptors = pickle.load(f)

def euclidean_distance(face_descriptor1, face_descriptor2):
    return np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))

# Initialize camera and queue
video_capture = cv2.VideoCapture(0)
frame_queue = queue.Queue(maxsize=5)

# Camera resolution
frame_width = 240
frame_height = 180
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Stop threads flag
stop_threads = threading.Event()

# Biến lưu trữ tên đối tượng đã xuất gần nhất
last_recognized_name = None

# Thread for reading frames
def read_frames():
    while not stop_threads.is_set():
        ret, frame = video_capture.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            frame_queue.get()
            frame_queue.put(frame)

# Thread for processing frames
def process_frames():
    global last_recognized_name
    while not stop_threads.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            faces = detector(frame)
            
            for face in faces:
                shape = shape_predictor(frame, face)
                face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)
                
                # Face recognition
                name = "Unknown"
                min_distance = 0.6
                for known_name, known_descriptor in known_face_descriptors:
                    distance = euclidean_distance(face_descriptor, known_descriptor)
                    if distance < min_distance:
                        min_distance = distance
                        name = known_name
                
                # Fetch data from Firebase or create new data if not found
                if name != "Unknown" and name != last_recognized_name:
                    last_recognized_name = name
                    data = ref.child(name).get()
                    
                    if data:
                        # Save data to a file
                        output_file = f"{name}_info.txt"
                        with open(output_file, "w", encoding="utf-8") as file:
                            # Print data in a table-like format
                            file.write("+---------+-----------------------+\n")
                            file.write("| Field   | Value                 |\n")
                            file.write("+---------+-----------------------+\n")
                            for key, value in data.items():
                                file.write(f"| {key:<8} | {value:<20} |\n")
                            file.write("+---------+-----------------------+\n")
                        print(f"Thông tin của '{name}' đã được lưu vào file {output_file}.")
                    else:
                        print(f"Không tìm thấy thông tin của '{name}' trong Firebase.")
                        # Nhập thông tin mới
                        print("Nhập thông tin mới:")
                        new_data = {}
                        new_data["age"] = input("Tuổi: ")
                        new_data["email"] = input("Email: ")
                        new_data["phone"] = input("Số điện thoại: ")
                        ref.child(name).set(new_data)
                        print(f"Thông tin mới cho '{name}' đã được lưu vào Firebase.")

                        # Lưu thông tin mới vào file
                        output_file = f"{name}_info.txt"
                        with open(output_file, "w", encoding="utf-8") as file:
                            # In dữ liệu mới vào bảng
                            file.write("+---------+-----------------------+\n")
                            file.write("| Field   | Value                 |\n")
                            file.write("+---------+-----------------------+\n")
                            for key, value in new_data.items():
                                file.write(f"| {key:<8} | {value:<20} |\n")
                            file.write("+---------+------------------------+\n")
                        print(f"Thông tin mới cho '{name}' đã được lưu vào file {output_file}.")

                # Draw rectangle and name on the frame
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Face Recognition", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_threads.set()
                break

# Start threads
read_thread = threading.Thread(target=read_frames)
process_thread = threading.Thread(target=process_frames)
read_thread.start()
process_thread.start()

# Wait for threads to finish
read_thread.join()
process_thread.join()

# Release resources
video_capture.release()
cv2.destroyAllWindows()

# ===============================================================================================
