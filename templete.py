import cv2
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from threading import Thread, Lock
from queue import Queue

# 初始化参数
num_states = 5
window_size = 30
feature_dim = 128

# 视频流输入队列和处理结果队列
frame_queue = Queue(maxsize=10)
result_queue = Queue(maxsize=10)

# 线程锁
lock = Lock()

def feature_extraction_worker():
    scaler = StandardScaler()
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        # 特征提取（使用预训练的轻量级CNN）
        feature = extract_feature(frame)
        feature_scaled = scaler.transform([feature])
        with lock:
            result_queue.put(feature_scaled)
        frame_queue.task_done()

def hmm_inference_worker():
    model = hmm.GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=10)
    window = []
    while True:
        feature = result_queue.get()
        if feature is None:
            break
        window.append(feature)
        if len(window) > window_size:
            window.pop(0)
        if len(window) == window_size:
            model.fit(window)
            log_likelihood = model.score_samples(window)
            keyframe = np.argmax(log_likelihood)
            # 输出关键帧
            print(f"Keyframe at position: {keyframe}")
        result_queue.task_done()

def extract_feature(frame):
    # 模拟特征提取过程
    return np.random.rand(feature_dim)

def main():
    cap = cv2.VideoCapture(0)  # 从摄像头捕获视频
    Thread(target=feature_extraction_worker).start()
    Thread(target=hmm_inference_worker).start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            continue
        frame_queue.put(frame)

    cap.release()
    frame_queue.put(None)
    result_queue.put(None)
    frame_queue.join()
    result_queue.join()

if __name__ == "__main__":
    main()
