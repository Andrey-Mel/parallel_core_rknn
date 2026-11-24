import cv2
import sys
import threading
import time
import json
import os
import numpy as np
from queue import Queue, Empty
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from rknnlite.api import RKNNLite

# realpath = os.path.abspath(__file__) #/home/radxa/VTOLL/detect_VTOL.py
# sep = os.path.sep
# realpath = realpath.split(sep)
# sys.path.append(os.path.join(realpath[0] + sep, *realpath[1:realpath.index('VTOLL') + 1]))

class ProcessingStatus(Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'

@dataclass
class FrameData:
    frame_id: int
    frame: np.ndarray
    timestamp: float
    status: ProcessingStatus = ProcessingStatus.PENDING
    detections: List[Any] = None
    error: str = None
    x_offset: int = 0
    y_offset: int = 0
    scale: float = 1.0

@dataclass
class ProcessorConfig:
    model_path: str
    num_workers: int
    input_size: Tuple[int, int] = (640, 640)
    frame_queue_size: int = 20
    result_queue_size: int = 50
    batch_size: int = 1
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45  #                                      !!! need chane
    max_fps: float = 30.0
    enable_profiling: bool = True
    classes: List[str] = ('people', 'car')

class RKNNProfiler:
    def __init__(self):
        self.inference_times = []
        self.preprocess_times = []
        self.postporocess_times = []
        self.frame_counts = {}

    def start_inference(self):
        return time.perf_counter()

    def end_inference(self, start_time, frame_id):
        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000
        self.inference_times.append(inference_time)

        if frame_id not in self.frame_counts:
            self.frame_counts[frame_id] = 0
        self.frame_counts[frame_id] += 1

        return inference_time

    def get_stats(self):
        if not self.inference_times:
            return {}

        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_frames': len(self.frame_counts),
            'total_inferences': len(self.inference_times)
        }

class EnhancedRKNNPool:
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.pool = Queue(maxsize=config.num_workers)
        self.available_events = {}
        self.lock = threading.RLock()
        self.models_created = 0

        # initializing models with diferent cores NPU
        self._initialize_models()

    def _initialize_models(self):
        core_masks = [
            RKNNLite.NPU_CORE_0,
            RKNNLite.NPU_CORE_1,
            RKNNLite.NPU_CORE_2,
            RKNNLite.NPU_CORE_AUTO
        ]

        for i in range(self.config.num_workers):
            core_mask = core_masks[i % len(core_masks)] if i < 3 else RKNNLite.NPU_CORE_AUTO

            try:
                rknn = RKNNLite()

                #load model
                ret = rknn.load_rknn(self.config.model_path)
                if ret != 0:
                    raise Exception(f'Load RKNN model failed wirh error code: {ret}')

                # initialize runtime with core
                ret = rknn.init_runtime(core_mask=core_mask)
                if ret != 0:
                    raise Exception(f'Init runtime failed with error code: {ret}')

                #save info about core
                rknn.core_id = i                                                        # ???
                rknn.core_mask = core_mask

                self.pool.put(rknn)
                self.models_created += 1
                print(f'Initialized RKNN model {i} on core {core_mask}')
            
            except Exception as e:
                print(f'Failed to initialize model {i}: {e}')
                if i == 0:                                                              #????
                    raise

    def get(self, timeout: float = 10.0) -> Tuple[RKNNLite, int]:
        """ get model from pool """
        try:
            rknn = self.pool.get(timeout=timeout)
            with self.lock:
                core_id = rknn.core_id                                                  #????
                self.available_events[core_id] = threading.Event()
            return rknn, core_id
        except Empty:
            raise TimeoutError('Timeout waiting for available RKNN model')

    def put(self, rknn: RKNNLite):
        """return model in pool"""
        with self.lock:
            core_id = rknn.core_id
            if core_id in self.available_events:
                del self.available_events[core_id]
            self.pool.put(rknn)

    
    def get_available_cores(self) -> List[int]:
        """ get list available core """
        available_cores = []
        with self.lock:
            for core_id in range(self.models_created):
                if core_id not in self.available_events:
                    available_cores.append(core_id)
        return available_cores

    def close(self):
        """release resourses"""
        while not self.pool.empty():
            try:
                rknn = self.pool.get_nowait()
                rknn.release()
            except Empty:
                break

class ParallelVideoProcessor:
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.rknn_pool = EnhancedRKNNPool(config)

        # Queue for recive data
        self.frame_queue = Queue(maxsize=config.frame_queue_size)
        self.result_queue = Queue(maxsize=config.result_queue_size)
        self.control_queue = Queue()

        # workers and manange
        self.workers: List[threading.Thread] = []
        self.running = False
        self.profiler = RKNNProfiler() if config.enable_profiling else None

        # statistic
        self.stats = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'worker_error': 0,
            'start_time': None,
            'last_frame_time': None
        }

        # callback for results
        self.detection_callbacks = []
        self.error_callbacks = []

    def add_detection_callback(self, callback):
        self.detection_callbacks.append(callback)

    def add_error_callback(self, callback):
        self.error_callbacks.append(callback)

    def _notify_detection(self, frame_data: FrameData):
        for callback in self.detection_callbacks:
            try:
                callback(frame_data)
            except Exception as e:
                print(f'Detection callback error: {e}')

    def _notify_error(self, frame_data: FrameData, error: str):
        for callback in self.error_callbacks:
            try:
                callback(frame_data, error)
            except Exception as e:
                print(f'Error callback error: {e}')

    def start_workers(self):
        """Start workers threads with processing error"""
        self.running = True
        self.stats['start_time'] = time.time()

        for i in range(self.config.num_workers):
            worker = threading.Thread(
                target = self._worker_loop,
                args=(i,),
                name=f'RKNN-WORKER-{i}',
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        # Start monitoring efficiency
        if self.profiler:
            monitor_thread = threading.Thread(
                target=self._monitor_performance,
                daemon=True
            )
            monitor_thread.start()

    def _worker_loop(self, worker_id: int):
        """Base worker with full processing Errors"""
        worker_stats = {
            "frames_processed": 0,
            "total_inference_time": 0,
            "last_activity": time.time()
        }

        while self.running:
            try:
                # get cadr
                try:
                    frame_data = self.frame_queue.get(timeout=1.0)
                    if frame_data is None:
                        break
                except Empty:
                    continue

                # check fps ogranichenie
                # if self._should_skip_frame():
                #     self.stats['frames_skipped'] += 1
                    # continue

                frame_data.status = ProcessingStatus.PROCESSING
                start_time = time.perf_counter()

                # get model from pool
                rknn, core_id = self.rknn_pool.get()

                try:
                    # detect, preprocess
                    preprocess_start = time.perf_counter()
                    input_data, x_offset, y_offset, scale = self.preprocess_frame(frame_data.frame)   
                    frame_data.x_offset = x_offset
                    frame_data.y_offset = y_offset
                    frame_data.scale = scale                     #<---!!!!!!
                    preprocess_time = (time.perf_counter() - preprocess_start) * 1000

                    inference_start = time.perf_counter()
                    outputs = rknn.inference(inputs=[input_data])
                    inference_time = self.profiler.end_inference(inference_start, frame_data.frame_id) if self.profiler else 0

                    postprocess_start = time.perf_counter()
                    detections = self.postprocess_outputs(outputs, frame_data.frame.shape)
                    
                    postprocess_time = (time.perf_counter() - postprocess_start) * 1000

                    # updata statistic
                    worker_stats['frames_processed'] += 1
                    worker_stats['total_inference_time'] += inference_time
                    worker_stats['last_activity'] = time.time()

                    # update data cadr
                    frame_data.detections = detections
                    frame_data.status = ProcessingStatus.COMPLETED

                    #send result
                    self.result_queue.put(frame_data)

                    # logging efficiency
                    if self.profiler:
                        self.profiler.preprocess_times.append(preprocess_time)
                        self.profiler.postporocess_times.append(postprocess_time)

                except Exception as e:
                    frame_data.status = ProcessingStatus.FAILED
                    frame_data.error = str(e)
                    self.stats['worker_error'] += 1
                    self._notify_error(frame_data, str(e))
                    print(f'Worker {worker_id} processing error: {e}')

                finally:
                    # return model rknn in pool
                    self.rknn_pool.put(rknn)
                    processing_time = (time.perf_counter() - start_time) * 1000

                    # loging eficiency
                    if worker_stats['frames_processed'] % 10 == 0:
                        print(f'Worker {worker_id} processed {worker_stats["frames_processed"]} frames'
                        f'avg time: {worker_stats["total_inference_time"]/worker_stats["frames_processed"]:.2f} ms')

            except Exception as e:
                print(f'Worker {worker_id} critical error: {e}')
                self.stats['worker_error'] += 1
                time.time(0.1) # protected from busy loop in case of errors
    
    def _should_skip_frame(self):
        """ Check to need skip cadr for control fps """
        if self.config.max_fps <= 0:
            return False
        
        current_time = time.time()
        if self.stats['last_frame_time'] is None:
            self.stats['last_frame_time'] = current_time
            return False

        min_frame_interval = 1.0 / self.config.max_fps
        time_since_last_frame = current_time - self.stats['last_frame_time']

        if time_since_last_frame < min_frame_interval:
            return True

        self.stats['last_frame_time'] = current_time
        return False

    def _monitor_performance(self):
        """Moitoring performans """
        while self.running:
            time.sleep(5.0) # 5 sec
            if self.profiler:
                stats = self.profiler.get_stats()
                queue_sizes = {
                    'frame_queue': self.frame_queue.qsize(),
                    'result_queue': self.result_queue.qsize(),
                    'available_workers': len(self.rknn_pool.get_available_cores())
                }
                print(f'Perfomance stats: {stats}')
                print(f'Queue stats: {queue_sizes}')

    def preprocess_frame(self, frame: np.ndarray):
        """ Preprocess frame for input model """
        h, w = frame.shape[:2]
        input_h, input_w = self.config.input_size

        scale = min(input_h / h, input_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # create canvas with padding
        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        y_offset = (input_h - new_h) // 2
        x_offset = (input_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        input_data = canvas.astype(np.float32)
        input_data = input_data.reshape(1, *input_data.shape)
        
        return input_data, x_offset, y_offset, scale                        # output x_ofset, y_offset, scale


    # POSTPROCESS
    def filter_boxes(self, boxes, box_confidences, box_class_probs): #(8400, 4), (8400, 1) - value=1, (8400, 4) - value : - 3 .... -24
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1) # 8400
        candidate, class_num = box_class_probs.shape # candidate = 8400, class_num = 4

        class_max_score = np.max(box_class_probs, axis=-1) # 8400
        classes = np.argmax(box_class_probs, axis=-1) # 8400 value show norm

        _class_pos = np.where(class_max_score* box_confidences >= self.config.confidence_threshold) # otricatelnie value(class_max_scores) * 1.0 (scores) = otricatelnie value
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes, scores, nms_thresh: float):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def dfl(self,position):
        # Distribution Focal Loss (DFL)
        # import torch
        def softmax(x, axis=-1):
            x_max = np.max(x, axis=axis, keepdims=True)
            x_shifted = x - x_max
            exp_x = np.exp(x_shifted)
            sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
            return exp_x / sum_exp
        # x = torch.tensor(position)
        x = np.array(position)  # !
        n,c,h,w = x.shape
        p_num = 4
        mc = c//p_num  # 16 = 64 / 4
        y = x.reshape(n,p_num,mc,h,w) # 1x4x16x80x80
        # y = y.softmax(2)
        y = softmax(y, axis=2) # 1x4x16x80x80  probabilyty axex 2 - 16
        # acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
        acc_metrix = np.array(range(mc)).astype(np.float32).reshape(1,1,mc,1,1) 
        y = (y*acc_metrix).sum(2) # 1x4x80x80  

        return y #y.numpy()

    def box_process(self,position):
        p = position.copy() # 1, 64, 80, 80
        # print('SHAPE POSITION: ', type(p), p.shape)
        input_h, input_w = self.config.input_size
        grid_h, grid_w = position.shape[2:4]  #  80, 80 single digits
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))  #setka 80x80
        col = col.reshape(1, 1, grid_h, grid_w) # 1x1x80x80
        row = row.reshape(1, 1, grid_h, grid_w) # 1x1x80x80
        grid = np.concatenate((col, row), axis=1) # 1x2x80x80
        stride = np.array([input_h//grid_h, input_w//grid_w]).reshape(1,2,1,1) #1x2x1x1

        position = self.dfl(position) # 1x4x80x80
        box_xy  = grid + 0.5 - position[:,0:2,:,:] # 1x2x80x80
        box_xy2 = grid + 0.5 + position[:,2:4,:,:] # 1x2x80x80
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1) # 1x4x80x80
        
        return xyxy

    def postprocess_outputs(self, outputs: List[np.ndarray], original_shape: Tuple[int, int])-> List[Dict]:
        detections = []
        boxes, scores, classes_conf = [], [], []
        defualt_branch=3 # default 3
        # print('LEN INPUT DATA: ', input_data[5][:,:,:].shape) # norm 9
        pair_per_branch = len(outputs)//defualt_branch

        # Python 忽略 score_sum 输出

        for i in range(defualt_branch):

            boxes.append(self.box_process(outputs[pair_per_branch * i]))  # input_data[pair_per_branch*i]
            classes_conf.append(outputs[pair_per_branch * i + 1]) # input_data[pair_per_branch*i+1]
            scores.append(np.ones_like(outputs[pair_per_branch * i + 1][:,:1,:,:], dtype=np.float32)) # input_data[pair_per_branch*i+1]
            
        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s, self.config.nms_threshold)

            if len(keep) != 0:
                # nboxes.append(b[keep])
                # nclasses.append(c[keep])
                # nscores.append(s[keep])
                for idx in keep:
                    x1, y1, x2, y2 = b[idx]
                    conf = s[idx]
                    cls_id = c[idx]
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': f'class_{self.config.classes[cls_id]}'
                    }
                    detections.append(detection)

        if not nclasses and not nscores:
            return detections

        # boxes = np.concatenate(nboxes)
        # classes = np.concatenate(nclasses)
        # scores = np.concatenate(nscores)

        return detections

    
    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """Base method processing video and output results"""
        if os.path.exists(video_path):
            print("Video OK")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f'Cannot open video: {video_path}')

        # processing write video
        writer = None
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        frame_id = 0
        pending_frames = {}

        #start workers
        self.start_workers()

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print('NOT READ CADR')
                    # self.running = False
                    break

                frame_data = FrameData(
                    frame_id=frame_id,
                    frame=frame.copy(),
                    timestamp = time.time()
                )
                try:
                    self.frame_queue.put(frame_data, timeout=1.0)  # or block=False if not want blocks

                except:
                    print("Frame queue full, skipping frame")
                    continue

                pending_frames[frame_id]=frame_data
                frame_id += 1
                self._process_results(writer, pending_frames)

                #Delete old cadrs
                if len(pending_frames) > 30:
                    oldest_frame_id = min(pending_frames.keys())
                    del pending_frames[oldest_frame_id]

            # processing need cadr
            # while pending_frames and self.running:                                                        #pending_frames and   !!!!!!!!!!!!!!!!!!!!!!!!
            #     self._process_results(writer, pending_frames)
        
        finally:
            # stop
            self.running = False

            #stop workers
            for _ in range(len(self.workers)):
                self.frame_queue.put(None)

            for worker in self.workers:
                worker.join()

            cap.release()
            cv2.destroyAllWindows()
            if writer:
                writer.release()
            self.rknn_pool.close()

            total_time = time.time() - self.stats['start_time']

            #final statistic
            print(f'Processing completed. Total time: {total_time:.2f} sec')
            print(f'Frames processed: {self.stats["frames_processed"]}')
            print(f'Frames skipped: {self.stats["frames_skipped"]}')
            print(f'Worker errors: {self.stats["worker_error"]}')

    def _process_results(self, writer, pending_frames: Dict[int, FrameData]):
        """Processed results from queue"""
        
        try:
            while True:
                frame_data = self.result_queue.get_nowait()
                self.stats["frames_processed"] += 1

                #delete waiting frames
                if frame_data.frame_id in pending_frames:
                    del pending_frames[frame_data.frame_id]

                # visual and write video
                if writer and frame_data.status == ProcessingStatus.COMPLETED and frame_data.detections is not None:
                    frame_with_detections = self._draw_detections(
                        frame_data.frame, 
                        frame_data.detections, 
                        frame_data
                    )
                    writer.write(frame_with_detections)

                    cv2.imshow('video', frame_with_detections)  
                    if cv2.waitKey(1) & 0xFF==ord('q'):
                        self.running = False 
                        break    
                    
                elif frame_data.status == ProcessingStatus.FAILED:
                    print(f'frame {frame_data.frame_id} failed: {frame_data.error}')                       #!!!!!!

        except Empty:
            # print(f'EMPTY results from queue RESULT_QUEUE')
            pass

    def _draw_detections(self, frame: np.ndarray, detections: List[Dict], frame_data: FrameData) -> np.ndarray:
        """Show detection on a cadr"""
        result_frame = frame.copy()
        x_offset = frame_data.x_offset
        y_offset = frame_data.y_offset
        scale = frame_data.scale
        for detection in detections:
            box = detection['bbox']
            x1, y1, x2, y2 = map(int, box)

            x1 = int((x1 - x_offset) / scale)
            y1 = int((y1 - y_offset) / scale)
            x2 = int((x2 - x_offset) / scale)
            y2 = int((y2 - y_offset) / scale)
            confidence = detection['confidence']
            class_name = detection['class_name']

            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = f'{class_name}_{confidence:.2f}'
            # label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return result_frame



# MAIN
def main():
    config = ProcessorConfig(
        model_path='/home/radxa/Desktop/N_models2cls_rknn_6out/best_yolo2cls_640.rknn', #best_sdu_v3_6out.rknn',best_yolo2cls_640.rknn
        num_workers=3,
        input_size=(640, 640),
        frame_queue_size = 2,
        result_queue_size=25,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        max_fps=30.0,
        enable_profiling=True
    )

    processor = ParallelVideoProcessor(config)

    # Add callbacks
    # def detection_callback(frame_data):
    #     if frame_data.detections:
    #         print(f'Frame {frame_data.frame_id}: {len(frame_data.detections)} detections')

    # def error_callbacks(frame_data, error):
    #     print(f"Error processing frame {frame_data.frame_id}: {error}")


    try:
        path_video = '/home/radxa/Desktop/N_models2cls_rknn_6out/palace.mp4'
        path_out_video = '/home/radxa/Desktop/N_models2cls_rknn_6out/out/test_palace_1280_720_demo.mp4'
        processor.process_video(path_video, path_out_video)

    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f'Processing failed: {e}')


if __name__ == '__main__':
    main()

