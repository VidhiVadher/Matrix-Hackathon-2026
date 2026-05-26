import os
import cv2
import time
import numpy as np
import threading
import logging
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# Ensure FFmpeg prefers UDP for low-latency RTSP and drops stale frames
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Optional PTZ (ONVIF)
# -----------------------------
USE_PTZ = True
try:
    if USE_PTZ:
        from onvif import ONVIFCamera
except ImportError as e:
    logger.error(f"[PTZ] ONVIF import failed: {e}. PTZ features will be disabled.")
    USE_PTZ = False


# -----------------------------
# Configuration
# -----------------------------
CAMERAS = [
    {
        "id": 0,
        "name": "CamA",
        "rtsp": "rtsp://admin:Admin@123@192.168.1.126/unicaststream/2",
        "is_ptz": True,
        "ptz": {
            "ip": "192.168.1.126",
            "port": 80,
            "user": "admin",
            "password": "Admin@123",
            "preset": "Preset 1",
        },
    },
    {
        "id": 1,
        "name": "CamB_PTZ",
        "rtsp": "rtsp://admin:Admin@123@192.168.1.55/unicaststream/2",
        "is_ptz": True,
        "ptz": {
            "ip": "192.168.1.55",
            "port": 80,
            "user": "admin",
            "password": "Admin@123",
            "preset": "Preset 1",
        },
    },
]

# Detection & ReID params optimized for RTX 2050 (4GB VRAM + Tensor Cores)
YOLO_MODEL_NAME = "yolo11m.pt"
DEEPSORT_EMBEDDER = "clip_ViT-B/16" # Ensure OpenAI CLIP is installed, or fallback to "mobilenet"
CONF_THRES = 0.55                   # Increased to prevent false positive ghosts
MIN_BBOX_AREA = 2000                # Minimum pixel area (w*h) to consider it a valid person
GLOBAL_DIST_THRESH = 0.18           # Tightened to prevent false ID matching across cameras
SHOW_WINDOW = True

# Global ReID recovery / safety gates
RECENT_TRACK_TTL_SEC = 2.5
RECENT_TRACK_SAME_CAM_DIST = 0.20
RECENT_TRACK_CROSS_CAM_DIST = 0.28
GLOBAL_EMBED_UPDATE_ALPHA = 0.05
GLOBAL_MATCH_MARGIN = 0.02
DISALLOW_SAME_CAM_DUP_GID = True

# Target lock / handoff config
HANDOFF_CONFIRM_FRAMES = 10         # Increased to require more stable tracking before switching cameras
SELECTED_TRACK_LOST_TIMEOUT = 7.0
RETURN_TO_PRESET_AFTER_IDLE = False
RETURN_TO_PRESET_IDLE_SEC = 10.0

# PTZ stabilization config - Tuned for smoother movements and less jitter
PTZ_SMOOTHING_ALPHA = 0.10          # Lowered: heavier smoothing on bounding box jitter
PTZ_DEAD_ZONE_X = 0.20              # Widened: larger center box where camera won't move
PTZ_DEAD_ZONE_Y = 0.20
PTZ_OUTER_ZONE_X = 0.40
PTZ_OUTER_ZONE_Y = 0.40
PTZ_MIN_CMD_INTERVAL = 0.25         # Prevent network spam to camera
PTZ_STOP_AFTER_STABLE_SEC = 0.4
PTZ_MAX_SPEED = 0.25
PTZ_MIN_SPEED = 0.05
PTZ_LOST_STOP_SEC = 0.2
PTZ_CMD_SMOOTHING_ALPHA = 0.40      # Lowered: smoother acceleration/deceleration
PTZ_MAX_ACCEL = 0.05
PTZ_MAX_STALE_FRAMES = 4

# -----------------------------
# Global selection state
# -----------------------------
selected_gid: Optional[int] = None
selected_cam_id: Optional[int] = None
selected_track_id: Optional[int] = None
selected_last_seen_ts: float = 0.0

state_lock = threading.Lock()

handoff_candidates = defaultdict(int)
visible_tracks = {}
window_names = {}
locked_target_bbox = {}


def clear_selected_target(ptz_clients=None, home_preset=None):
    global selected_gid, selected_cam_id, selected_track_id, selected_last_seen_ts
    global handoff_candidates, locked_target_bbox

    with state_lock:
        old_cam = selected_cam_id

        selected_gid = None
        selected_cam_id = None
        selected_track_id = None
        selected_last_seen_ts = 0.0
        handoff_candidates.clear()
        locked_target_bbox.clear()

    if ptz_clients is not None:
        for ptz in ptz_clients.values():
            ptz.reset_filter()
            ptz.stop()

    if home_preset is not None and old_cam is not None and ptz_clients is not None:
        ptz = ptz_clients.get(old_cam)
        if ptz is not None:
            ptz.goto_preset(home_preset)


# -----------------------------
# Robust RTSP Reader with Auto-Reconnect
# -----------------------------
class RTSPReader:
    def __init__(self, url: str, name: str):
        self.url = url
        self.name = name
        self.cap = None
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.reconnect_delay = 2.0

        self._connect()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _connect(self):
        if self.cap is not None:
            self.cap.release()
        logger.info(f"[CAM {self.name}] Connecting to RTSP stream...")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            logger.error(f"[CAM {self.name}] Cannot open RTSP: {self.url}")
        else:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            except Exception:
                pass
            logger.info(f"[CAM {self.name}] Stream connected successfully.")

    def _loop(self):
        failed_reads = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(self.reconnect_delay)
                self._connect()
                continue

            ret, frame = self.cap.read()
            if not ret:
                failed_reads += 1
                if failed_reads > 30: 
                    logger.warning(f"[CAM {self.name}] Stream dropped. Reconnecting...")
                    self.cap.release()
                    self.cap = None
                    failed_reads = 0
                time.sleep(0.01)
                continue
            
            failed_reads = 0
            with self.lock:
                self.latest = frame

    def read(self):
        with self.lock:
            frame = None if self.latest is None else self.latest.copy()
        return frame

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()


# -----------------------------
# PTZ Client
# -----------------------------
class PTZClient:
    def __init__(self, ip, port, user, password):
        self.ip = ip
        self.port = port
        self.user = user
        self.password = password

        self.cam = None
        self.media = None
        self.ptz = None
        self.profile = None

        self.last_cmd_ts = 0.0
        self.last_centered_ts = 0.0
        self.last_seen_target_ts = 0.0
        self.last_move_active = False

        self.last_pan_speed = 0.0
        self.last_tilt_speed = 0.0

        self.smooth_nx = None
        self.smooth_ny = None

    def _ensure(self):
        if self.cam is None:
            self.cam = ONVIFCamera(self.ip, self.port, self.user, self.password)
            self.media = self.cam.create_media_service()
            self.ptz = self.cam.create_ptz_service()
            self.profile = self.media.GetProfiles()[0]

    def goto_preset(self, preset_name: str):
        try:
            self._ensure()
            presets = self.ptz.GetPresets({"ProfileToken": self.profile.token})
            token = None
            for p in presets:
                if getattr(p, "Name", "") == preset_name:
                    token = p.token
                    break
            if token is None:
                logger.warning(f"[PTZ] Preset '{preset_name}' not found on {self.ip}")
                return
            req = self.ptz.create_type("GotoPreset")
            req.ProfileToken = self.profile.token
            req.PresetToken = token
            self.ptz.GotoPreset(req)
            logger.info(f"[PTZ] {self.ip} -> GotoPreset({preset_name})")
        except Exception as e:
            logger.error(f"[PTZ] GotoPreset error: {e}")

    def stop(self):
        if not self.last_move_active:
            return # Prevent spamming stop command
            
        try:
            self._ensure()
            self.ptz.Stop({
                "ProfileToken": self.profile.token,
                "PanTilt": True,
                "Zoom": True
            })
            self.last_move_active = False
            self.last_pan_speed = 0.0
            self.last_tilt_speed = 0.0
        except Exception as e:
            logger.error(f"[PTZ] Stop error: {e}")

    def _continuous_move(self, pan_speed: float, tilt_speed: float):
        try:
            self._ensure()
            req = self.ptz.create_type("ContinuousMove")
            req.ProfileToken = self.profile.token
            req.Velocity = {
                "PanTilt": {
                    "x": float(pan_speed),
                    "y": float(tilt_speed),
                }
            }
            self.ptz.ContinuousMove(req)
            self.last_move_active = True
        except Exception as e:
            logger.error(f"[PTZ] ContinuousMove error: {e}")

    def track_target(self, nx: float, ny: float, now: float = None):
        if now is None:
            now = time.time()

        try:
            self._ensure()
        except Exception as e:
            logger.error(f"[PTZ] init error: {e}")
            return

        self.last_seen_target_ts = now

        if self.smooth_nx is None or self.smooth_ny is None:
            self.smooth_nx = nx
            self.smooth_ny = ny
        else:
            self.smooth_nx = (1 - PTZ_SMOOTHING_ALPHA) * self.smooth_nx + PTZ_SMOOTHING_ALPHA * nx
            self.smooth_ny = (1 - PTZ_SMOOTHING_ALPHA) * self.smooth_ny + PTZ_SMOOTHING_ALPHA * ny

        ex = self.smooth_nx - 0.5
        ey = self.smooth_ny - 0.5

        abs_ex = abs(ex)
        abs_ey = abs(ey)

        inside_dead_x = abs_ex <= PTZ_DEAD_ZONE_X / 2
        inside_dead_y = abs_ey <= PTZ_DEAD_ZONE_Y / 2

        if inside_dead_x and inside_dead_y:
            if self.last_centered_ts == 0:
                self.last_centered_ts = now

            if self.last_move_active and (now - self.last_centered_ts) >= PTZ_STOP_AFTER_STABLE_SEC:
                self.stop()
            return
        else:
            self.last_centered_ts = 0.0

        if (now - self.last_cmd_ts) < PTZ_MIN_CMD_INTERVAL:
            return

        def compute_speed(err, dead_zone, outer_zone):
            aerr = abs(err)
            if aerr <= dead_zone / 2:
                return 0.0

            start = dead_zone / 2
            end = outer_zone / 2
            if end <= start:
                end = start + 1e-6

            if aerr >= end:
                mag = PTZ_MAX_SPEED
            else:
                frac = (aerr - start) / (end - start)
                mag = PTZ_MIN_SPEED + frac * (PTZ_MAX_SPEED - PTZ_MIN_SPEED)

            return mag if err > 0 else -mag

        pan_speed = compute_speed(ex, PTZ_DEAD_ZONE_X, PTZ_OUTER_ZONE_X)
        tilt_speed = compute_speed(-ey, PTZ_DEAD_ZONE_Y, PTZ_OUTER_ZONE_Y)

        def smooth_limit(desired, last):
            smoothed = (1 - PTZ_CMD_SMOOTHING_ALPHA) * last + PTZ_CMD_SMOOTHING_ALPHA * desired
            delta = smoothed - last

            if delta > PTZ_MAX_ACCEL:
                smoothed = last + PTZ_MAX_ACCEL
            elif delta < -PTZ_MAX_ACCEL:
                smoothed = last - PTZ_MAX_ACCEL

            if desired == 0.0 and abs(smoothed) < PTZ_MIN_SPEED:
                smoothed = 0.0

            if smoothed > PTZ_MAX_SPEED:
                smoothed = PTZ_MAX_SPEED
            elif smoothed < -PTZ_MAX_SPEED:
                smoothed = -PTZ_MAX_SPEED

            return smoothed

        pan_speed = smooth_limit(pan_speed, self.last_pan_speed)
        tilt_speed = smooth_limit(tilt_speed, self.last_tilt_speed)

        if pan_speed == 0.0 and tilt_speed == 0.0:
            if self.last_move_active:
                self.stop()
            return

        self._continuous_move(pan_speed, tilt_speed)
        self.last_pan_speed = pan_speed
        self.last_tilt_speed = tilt_speed
        self.last_cmd_ts = now

    def on_target_lost(self, now: float = None, preset_name: str = None):
        if now is None:
            now = time.time()

        if self.last_seen_target_ts == 0:
            return

        if self.last_move_active and (now - self.last_seen_target_ts) >= PTZ_LOST_STOP_SEC:
            self.stop()

        if RETURN_TO_PRESET_AFTER_IDLE and preset_name is not None:
            if (now - self.last_seen_target_ts) >= RETURN_TO_PRESET_IDLE_SEC:
                self.goto_preset(preset_name)
                self.last_seen_target_ts = now

    def reset_filter(self):
        self.smooth_nx = None
        self.smooth_ny = None
        self.last_centered_ts = 0.0
        self.last_pan_speed = 0.0
        self.last_tilt_speed = 0.0


# -----------------------------
# Global ReID manager
# -----------------------------
def cosine_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return 1.0 - np.dot(a, b)


def bbox_center_and_area(ltrb):
    l, t, r, b = ltrb
    cx = (l + r) / 2.0
    cy = (t + b) / 2.0
    area = max(1.0, (r - l) * (b - t))
    return cx, cy, area


def normalized_center_distance(b1, b2, frame_w, frame_h):
    c1x, c1y, _ = bbox_center_and_area(b1)
    c2x, c2y, _ = bbox_center_and_area(b2)
    dx = abs(c1x - c2x) / max(1.0, frame_w)
    dy = abs(c1y - c2y) / max(1.0, frame_h)
    return float(np.sqrt(dx * dx + dy * dy))


def area_ratio(b1, b2):
    _, _, a1 = bbox_center_and_area(b1)
    _, _, a2 = bbox_center_and_area(b2)
    return float(min(a1, a2) / max(a1, a2))


class GlobalReIDManager:
    def __init__(self, dist_thresh=0.28):
        self.dist_thresh = dist_thresh
        self.global_id_counter = 1

        self.global_embeddings = {}
        self.global_last_cam = {}
        self.track_to_global = {}
        self.global_active_tracks = {}

        self.recent_lost_tracks = []

        self.track_last_bbox = {}
        self.track_last_feat = {}
        self.track_last_seen_ts = {}

    def cleanup_missing_tracks(self, active_track_keys, now):
        active_track_keys = set(active_track_keys)
        old_keys = list(self.track_to_global.keys())

        for key in old_keys:
            if key in active_track_keys:
                continue

            gid = self.track_to_global[key]
            bbox = self.track_last_bbox.get(key)
            feat = self.track_last_feat.get(key)
            last_seen = self.track_last_seen_ts.get(key, now)

            self.recent_lost_tracks.append({
                "key": key,
                "gid": gid,
                "cam_id": key[0],
                "tid": key[1],
                "feat": feat,
                "bbox": bbox,
                "last_seen": last_seen,
            })

            del self.track_to_global[key]

            if gid in self.global_active_tracks:
                self.global_active_tracks[gid].discard(key)
                if len(self.global_active_tracks[gid]) == 0:
                    del self.global_active_tracks[gid]

        self.recent_lost_tracks = [
            x for x in self.recent_lost_tracks
            if (now - x.get("last_seen", now)) <= RECENT_TRACK_TTL_SEC
        ]

    def _try_recover_recent_gid(self, cam_id, local_track_id, feat, bbox, frame_w, frame_h, now):
        best_item = None
        best_score = 999.0

        for item in self.recent_lost_tracks:
            lost_feat = item.get("feat")
            lost_bbox = item.get("bbox")
            if lost_feat is None or lost_bbox is None:
                continue

            dist = cosine_distance(feat, lost_feat)
            same_cam = (item.get("cam_id") == cam_id)

            if same_cam:
                cdist = normalized_center_distance(bbox, lost_bbox, frame_w, frame_h)
                ar = area_ratio(bbox, lost_bbox)

                if dist > RECENT_TRACK_SAME_CAM_DIST:
                    continue
                if cdist > 0.22:
                    continue
                if ar < 0.60:
                    continue

                score = dist + 0.5 * cdist + 0.2 * (1.0 - ar)
            else:
                if dist > RECENT_TRACK_CROSS_CAM_DIST:
                    continue
                score = dist

            if score < best_score:
                best_score = score
                best_item = item

        if best_item is None:
            return None, False

        gid = int(best_item["gid"])
        key = (cam_id, local_track_id)
        self.track_to_global[key] = gid
        self.global_active_tracks.setdefault(gid, set()).add(key)

        if gid in self.global_embeddings:
            old = self.global_embeddings[gid]
            self.global_embeddings[gid] = (1.0 - GLOBAL_EMBED_UPDATE_ALPHA) * old + GLOBAL_EMBED_UPDATE_ALPHA * feat
        else:
            self.global_embeddings[gid] = feat

        try:
            self.recent_lost_tracks.remove(best_item)
        except ValueError:
            pass

        return gid, True

    def _eligible_gid(self, gid, cam_id):
        active_tracks = self.global_active_tracks.get(gid, set())

        if len(active_tracks) >= 2:
            return False

        if DISALLOW_SAME_CAM_DUP_GID:
            for acam, _ in active_tracks:
                if acam == cam_id:
                    return False

        return True

    def _best_global_match(self, cam_id, feat):
        best_gid = None
        best_dist = 999.0
        second_best = 999.0

        for gid, gfeat in self.global_embeddings.items():
            if not self._eligible_gid(gid, cam_id):
                continue

            d = cosine_distance(gfeat, feat)
            if d < best_dist:
                second_best = best_dist
                best_dist = d
                best_gid = gid
            elif d < second_best:
                second_best = d

        return best_gid, best_dist, second_best

    def assign_or_create(self, cam_id, local_track_id, feat, bbox, frame_w, frame_h, now):
        key = (cam_id, local_track_id)

        if key in self.track_to_global:
            gid = self.track_to_global[key]
            old = self.global_embeddings.get(gid)
            if old is None:
                self.global_embeddings[gid] = feat
            else:
                self.global_embeddings[gid] = (1.0 - GLOBAL_EMBED_UPDATE_ALPHA) * old + GLOBAL_EMBED_UPDATE_ALPHA * feat
            return gid, False, False

        gid, recovered = self._try_recover_recent_gid(cam_id, local_track_id, feat, bbox, frame_w, frame_h, now)
        if gid is not None:
            return gid, True, True

        best_gid, best_dist, second_best = self._best_global_match(cam_id, feat)

        if best_gid is not None and best_dist < self.dist_thresh:
            if second_best < 998.0 and (second_best - best_dist) < GLOBAL_MATCH_MARGIN:
                best_gid = None
            else:
                self.track_to_global[key] = best_gid
                self.global_active_tracks.setdefault(best_gid, set()).add(key)
                g_old = self.global_embeddings[best_gid]
                self.global_embeddings[best_gid] = (1.0 - GLOBAL_EMBED_UPDATE_ALPHA) * g_old + GLOBAL_EMBED_UPDATE_ALPHA * feat
                return best_gid, True, False

        gid_new = self.global_id_counter
        self.global_id_counter += 1
        self.global_embeddings[gid_new] = feat
        self.track_to_global[key] = gid_new
        self.global_active_tracks.setdefault(gid_new, set()).add(key)
        return gid_new, True, False

    def update_track(self, cam_id, local_track_id, feat, bbox, frame_w, frame_h, now):
        gid, is_new_mapping, recovered = self.assign_or_create(
            cam_id, local_track_id, feat, bbox, frame_w, frame_h, now
        )

        key = (cam_id, local_track_id)
        self.track_last_bbox[key] = bbox
        self.track_last_feat[key] = feat
        self.track_last_seen_ts[key] = now

        prev_cam = self.global_last_cam.get(gid, None)
        moved = is_new_mapping and prev_cam is not None and prev_cam != cam_id
        if is_new_mapping:
            self.global_last_cam[gid] = cam_id

        return gid, prev_cam, moved, recovered


# -----------------------------
# Mouse callback
# -----------------------------
def make_mouse_callback(cam_id):
    def on_mouse(event, x, y, flags, param):
        global selected_gid, selected_cam_id, selected_track_id, selected_last_seen_ts
        global handoff_candidates, locked_target_bbox

        if event == cv2.EVENT_LBUTTONDOWN:
            with state_lock:
                tracks = list(visible_tracks.get(cam_id, []))

            best = None
            best_area = None
            for item in tracks:
                l, t, r, b = item["bbox"]
                if not (l <= x <= r and t <= y <= b):
                    continue

                area = max(1, (r - l) * (b - t))
                if best is None or area < best_area:
                    best = item
                    best_area = area

            if best is not None:
                gid = best["gid"]
                tid = best["tid"]
                l, t, r, b = best["bbox"]
                with state_lock:
                    selected_gid = gid
                    selected_cam_id = cam_id
                    selected_track_id = tid
                    selected_last_seen_ts = time.time()
                    locked_target_bbox.clear()
                    locked_target_bbox[cam_id] = (l, t, r, b)
                    handoff_candidates.clear()
                logger.info(f"[SELECT] Selected GID={gid}, CAM={cam_id}, TRACK={tid}")
    return on_mouse


# -----------------------------
# Helpers
# -----------------------------
def clear_stale_lock(now, ptz_clients):
    global selected_gid, selected_cam_id

    with state_lock:
        sel_gid = selected_gid
        sel_cam = selected_cam_id
        sel_last_seen = selected_last_seen_ts

    if sel_gid is None or sel_cam is None:
        return

    if (now - sel_last_seen) <= SELECTED_TRACK_LOST_TIMEOUT:
        return

    home_preset = None
    for cam_cfg in CAMERAS:
        if cam_cfg["id"] == sel_cam:
            home_preset = cam_cfg["ptz"].get("preset")
            break

    logger.info(f"[SELECT] Lost target G{sel_gid}; returning home and resuming detection")
    clear_selected_target(ptz_clients=ptz_clients, home_preset=home_preset)


def find_best_selected_gid_candidate(sel_gid, sel_cam):
    best_candidate = None
    best_score = 1e9

    for cam_id, items in visible_tracks.items():
        if cam_id == sel_cam:
            continue

        for item in items:
            if item.get("gid") != sel_gid:
                continue
            if item.get("tsu", 0) != 0:
                continue

            l, t, r, b = item["bbox"]
            area = max(1.0, (r - l) * (b - t))
            cx = (l + r) / 2.0
            cy = (t + b) / 2.0

            frame_center_penalty = abs(cx - 640) / 640.0 + abs(cy - 360) / 360.0
            score = frame_center_penalty - 0.00001 * area

            if score < best_score:
                best_score = score
                best_candidate = item

    return best_candidate


# -----------------------------
# Main
# -----------------------------
def main():
    global visible_tracks
    global selected_gid, selected_cam_id, selected_track_id, selected_last_seen_ts
    global handoff_candidates

    # Check CUDA Availability explicitly
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"System detected device: {device}")

    logger.info(f"Loading YOLO Model: {YOLO_MODEL_NAME}")
    model = YOLO(YOLO_MODEL_NAME)
    try:
        model.to(device)
        logger.info(f"[YOLO] Loaded successfully on {device}.")
    except Exception as e:
        logger.warning(f"[YOLO] Failed to load on {device}, falling back to CPU: {e}")
        device = "cpu"
        model.to("cpu")

    trackers = {}
    readers = {}
    ptz_clients = {}

    logger.info(f"Loading DeepSort Embedder: {DEEPSORT_EMBEDDER} with FP16 support")
    for cam_cfg in CAMERAS:
        cam_id = cam_cfg["id"]

        trackers[cam_id] = DeepSort(
            max_age=40,
            n_init=3,
            nms_max_overlap=1.0,
            max_iou_distance=0.85,
            max_cosine_distance=0.20,
            nn_budget=100,
            embedder=DEEPSORT_EMBEDDER, 
            embedder_gpu=(device != "cpu"),
            half=True, 
        )

        try:
            readers[cam_id] = RTSPReader(cam_cfg["rtsp"], cam_cfg["name"])
        except Exception as e:
            logger.error(f"[CAM] ERROR opening RTSP for {cam_cfg['name']}: {e}")
            continue

        if USE_PTZ and cam_cfg["is_ptz"]:
            ptz_cfg = cam_cfg["ptz"]
            ptz_clients[cam_id] = PTZClient(
                ptz_cfg["ip"],
                ptz_cfg["port"],
                ptz_cfg["user"],
                ptz_cfg["password"],
            )

    global_reid = GlobalReIDManager(dist_thresh=GLOBAL_DIST_THRESH)

    if SHOW_WINDOW:
        for cam_cfg in CAMERAS:
            cam_id = cam_cfg["id"]
            cam_name = cam_cfg["name"]
            win = f"{cam_name} (id {cam_id})"
            window_names[cam_id] = win
            cv2.namedWindow(win)
            cv2.setMouseCallback(win, make_mouse_callback(cam_id))

    logger.info("Starting processing loop. Press 'q' to quit, 'c' to clear selected target.")

    try:
        while True:
            now = time.time()
            any_ok = False

            with state_lock:
                visible_tracks = {}

            current_active_keys = []
            frame_cache = {}

            for cam_cfg in CAMERAS:
                cam_id = cam_cfg["id"]
                if cam_id not in readers:
                    continue

                frame = readers[cam_id].read()
                if frame is None:
                    continue
                frame_cache[cam_id] = frame

            per_cam_results = {}

            for cam_cfg in CAMERAS:
                cam_id = cam_cfg["id"]
                cam_name = cam_cfg["name"]

                if cam_id not in frame_cache:
                    continue

                frame = frame_cache[cam_id]
                any_ok = True
                h, w = frame.shape[:2]

                results = model(frame, conf=CONF_THRES, verbose=False, device=device, half=(device != "cpu"))[0]

                detections = []
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id != 0: # Only track persons
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_w, box_h = x2 - x1, y2 - y1
                    if box_w * box_h < MIN_BBOX_AREA: # Filter out tiny false positives
                        continue
                        
                    detections.append(([x1, y1, box_w, box_h], conf, cls_id))

                if selected_gid is not None and selected_cam_id == cam_id and cam_id in locked_target_bbox:
                    last_l, last_t, last_r, last_b = locked_target_bbox[cam_id]
                    last_cx = (last_l + last_r) / 2.0
                    last_cy = (last_t + last_b) / 2.0
                    last_area = max(1.0, (last_r - last_l) * (last_b - last_t))

                    max_dist = max(250.0, float(np.sqrt(last_area) * 2.5))

                    best_det = None
                    best_score = 1e9

                    for det_bbox, det_conf, det_cls_id in detections:
                        det_l, det_t, det_w, det_h = det_bbox
                        det_r = det_l + det_w
                        det_b = det_t + det_h

                        det_cx = (det_l + det_r) / 2.0
                        det_cy = (det_t + det_b) / 2.0
                        det_area = max(1.0, det_w * det_h)

                        dist = float(np.sqrt((det_cx - last_cx) ** 2 + (det_cy - last_cy) ** 2))
                        det_area_ratio = min(det_area, last_area) / max(det_area, last_area)

                        inter_l = max(det_l, last_l)
                        inter_t = max(det_t, last_t)
                        inter_r = min(det_r, last_r)
                        inter_b = min(det_b, last_b)
                        inter_w = max(0.0, inter_r - inter_l)
                        inter_h = max(0.0, inter_b - inter_t)
                        inter_area = inter_w * inter_h
                        union_area = det_area + last_area - inter_area
                        iou = (inter_area / union_area) if union_area > 0 else 0.0

                        if dist > max_dist and iou < 0.05:
                            continue
                        if det_area_ratio < 0.25:
                            continue

                        score = (dist / max_dist) - 0.35 * iou + 0.15 * (1.0 - det_area_ratio) - 0.05 * det_conf
                        if score < best_score:
                            best_score = score
                            best_det = (det_bbox, det_conf, det_cls_id)

                    detections = [best_det] if best_det is not None else []

                tracker = trackers[cam_id]
                tracks = tracker.update_tracks(detections, frame=frame)

                per_cam_results[cam_id] = {
                    "frame": frame,
                    "tracks": tracks,
                    "shape": (h, w),
                    "cam_name": cam_name,
                    "cam_cfg": cam_cfg,
                }

                for track in tracks:
                    if track.is_confirmed():
                        current_active_keys.append((cam_id, track.track_id))

            global_reid.cleanup_missing_tracks(current_active_keys, now)

            for cam_id, data in per_cam_results.items():
                frame = data["frame"]
                tracks = data["tracks"]
                h, w = data["shape"]
                cam_name = data["cam_name"]

                cam_visible = []

                with state_lock:
                    sel_gid = selected_gid
                    sel_cam = selected_cam_id
                    sel_tid = selected_track_id

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    tid = track.track_id
                    l, t, r, b = track.to_ltrb()
                    l, t, r, b = int(l), int(t), int(r), int(b)

                    feat = None
                    if hasattr(track, "features") and len(track.features) > 0:
                        feat = track.features[-1]
                    if feat is None:
                        continue

                    feat = np.array(feat, dtype=np.float32)
                    bbox = (l, t, r, b)

                    gid, prev_cam, moved, recovered = global_reid.update_track(
                        cam_id, tid, feat, bbox, w, h, now
                    )

                    item = {
                        "bbox": bbox,
                        "gid": gid,
                        "tid": tid,
                        "cam_id": cam_id,
                        "tsu": int(getattr(track, "time_since_update", 0)),
                    }
                    cam_visible.append(item)

                    is_exact_selected = (
                        sel_gid is not None and
                        sel_cam == cam_id and
                        sel_tid == tid
                    )

                    is_gid_match = (
                        sel_gid is not None and
                        sel_gid == gid
                    )

                    color = (0, 255, 0)
                    label = f"G{gid} C{cam_id} T{tid}"

                    if recovered:
                        label += " RECOVER"

                    if is_exact_selected:
                        color = (0, 0, 255)
                        label = f"*G{gid}* C{cam_id} T{tid}"
                        if recovered:
                            label += " RECOVER"

                        if int(getattr(track, "time_since_update", 0)) == 0:
                            with state_lock:
                                selected_last_seen_ts = now
                            try:
                                ol, ot, or_, ob = track.to_ltrb(orig=True)
                                with state_lock:
                                    locked_target_bbox[cam_id] = (int(ol), int(ot), int(or_), int(ob))
                            except Exception:
                                with state_lock:
                                    locked_target_bbox[cam_id] = bbox
                    elif is_gid_match:
                        color = (0, 165, 255)

                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (l, max(0, t - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    if moved and prev_cam is not None:
                        logger.info(
                            f"[HANDOFF] Global {gid} moved from {prev_cam} to {cam_id} "
                            f"({CAMERAS[prev_cam]['name']} -> {cam_name})"
                        )

                with state_lock:
                    visible_tracks[cam_id] = cam_visible

            with state_lock:
                sel_gid = selected_gid
                sel_cam = selected_cam_id
                sel_tid = selected_track_id
                sel_last_seen = selected_last_seen_ts

            if sel_gid is not None and sel_cam is not None and sel_tid is not None:
                exact_visible = any(
                    item.get("tid") == sel_tid and item.get("tsu", 0) == 0
                    for item in visible_tracks.get(sel_cam, [])
                )

                if exact_visible:
                    handoff_candidates.clear()
                else:
                    best_candidate = find_best_selected_gid_candidate(sel_gid, sel_cam)

                    stale_keys = []
                    for key in list(handoff_candidates.keys()):
                        cam_id, tid = key
                        found = any(
                            item.get("cam_id") == cam_id and item.get("tid") == tid and item.get("gid") == sel_gid and item.get("tsu", 0) == 0
                            for item in visible_tracks.get(cam_id, [])
                        )
                        if not found:
                            stale_keys.append(key)

                    for key in stale_keys:
                        handoff_candidates.pop(key, None)

                    if best_candidate is not None:
                        key = (best_candidate["cam_id"], best_candidate["tid"])
                        handoff_candidates[key] += 1

                        if handoff_candidates[key] >= HANDOFF_CONFIRM_FRAMES:
                            old_cam = sel_cam
                            new_cam = best_candidate["cam_id"]
                            new_tid = best_candidate["tid"]

                            with state_lock:
                                selected_cam_id = new_cam
                                selected_track_id = new_tid
                                selected_last_seen_ts = now
                                locked_target_bbox.clear()
                                locked_target_bbox[new_cam] = best_candidate["bbox"]
                                handoff_candidates.clear()

                            if old_cam in ptz_clients:
                                ptz_clients[old_cam].reset_filter()
                                ptz_clients[old_cam].stop()
                            if new_cam in ptz_clients:
                                ptz_clients[new_cam].reset_filter()

                            logger.info(f"[FOLLOW] Confirmed handoff G{sel_gid} -> camera {new_cam}, track {new_tid}")
                    else:
                        if (now - sel_last_seen) > SELECTED_TRACK_LOST_TIMEOUT:
                            handoff_candidates.clear()
                            clear_stale_lock(now, ptz_clients)

            for cam_cfg in CAMERAS:
                cam_id = cam_cfg["id"]

                if not (USE_PTZ and cam_cfg["is_ptz"] and cam_id in ptz_clients):
                    continue

                ptz = ptz_clients[cam_id]
                target_found_for_this_ptz = False

                with state_lock:
                    sel_gid = selected_gid
                    sel_cam = selected_cam_id
                    sel_tid = selected_track_id

                if sel_gid is not None and sel_cam == cam_id and sel_tid is not None:
                    item = None
                    for it in visible_tracks.get(cam_id, []):
                        if it.get("tid") == sel_tid and it.get("tsu", 0) <= PTZ_MAX_STALE_FRAMES:
                            item = it
                            break

                    if item is not None:
                        l, t, r, b = item["bbox"]
                        frame = per_cam_results[cam_id]["frame"]
                        h, w = frame.shape[:2]

                        cx = (l + r) / 2.0
                        cy = (t + b) / 2.0
                        nx = cx / w
                        ny = cy / h

                        ptz.track_target(nx, ny, now=now)
                        target_found_for_this_ptz = True

                if not target_found_for_this_ptz:
                    ptz.on_target_lost(now=now, preset_name=cam_cfg["ptz"].get("preset"))

            if SHOW_WINDOW:
                for cam_id, data in per_cam_results.items():
                    win = window_names[cam_id]
                    frame = data["frame"]

                    info_y = 25
                    cv2.putText(
                        frame,
                        f"Selected: G={selected_gid} CAM={selected_cam_id} T={selected_track_id}",
                        (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow(win, frame)

            if not any_ok:
                time.sleep(0.01)

            if SHOW_WINDOW:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit signal received.")
                    break
                elif key == ord("c"):
                    clear_selected_target(ptz_clients=ptz_clients, home_preset=None)
                    logger.info("[SELECT] Cleared selected target")

    except KeyboardInterrupt:
        logger.info("\n[EXIT] KeyboardInterrupt detected. Shutting down gracefully...")

    finally:
        # Cleanup Resources
        for reader in readers.values():
            reader.release()

        for ptz in ptz_clients.values():
            try:
                ptz.stop()
            except Exception:
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()