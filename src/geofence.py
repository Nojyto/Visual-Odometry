import cv2
import numpy as np


class ORBGeofence:
    def __init__(self, map_image_path, boundary_margin=100):
        self.full_map = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if self.full_map is None:
            raise ValueError("Could not load map image.")

        self.map_h, self.map_w = self.full_map.shape
        self.boundary_margin = boundary_margin

        self.orb_map = cv2.ORB_create(nfeatures=25000, scaleFactor=1.2, nlevels=8)
        self.orb_frame = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)

        print("Extracting features from the global map. This will take a few seconds...")
        self.kp_map, self.des_map = self.orb_map.detectAndCompute(self.full_map, None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def update_position(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray_frame.shape
        kp_frame, des_frame = self.orb_frame.detectAndCompute(gray_frame, None)

        if des_frame is None or len(des_frame) < 4:
            return None, None, "ERROR: BLURRY FRAME"

        matches = self.matcher.knnMatch(des_frame, self.des_map, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)

        MIN_MATCH_COUNT = 15

        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                return None, None, "ERROR: HOMOGRAPHY MATH FAILED"

            pts = np.float32([[0, 0], [0, frame_h - 1], [frame_w - 1, frame_h - 1], [frame_w - 1, 0]]).reshape(-1, 1, 2)
            camera_footprint = cv2.perspectiveTransform(pts, M)

            center_x = int(np.mean(camera_footprint[:, 0, 0]))
            center_y = int(np.mean(camera_footprint[:, 0, 1]))
            center_pos = (center_x, center_y)

            status = "SAFE"
            if (
                center_x < self.boundary_margin
                or center_x > self.map_w - self.boundary_margin
                or center_y < self.boundary_margin
                or center_y > self.map_h - self.boundary_margin
            ):
                status = "WARNING: OUT OF BOUNDS"

            return center_pos, np.int32(camera_footprint), status
        else:
            return None, None, f"TRACKING LOST: Only {len(good_matches)} matches"


SOURCE_MAP = "../sim/map.jpeg"
SOURCE_VIDEO = "../data/feed.mp4"

if __name__ == "__main__":
    geofence = ORBGeofence(SOURCE_MAP)
    cap = cv2.VideoCapture(SOURCE_VIDEO)
    display_map = cv2.imread(SOURCE_MAP)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pos, footprint, status = geofence.update_position(frame)

        viz_map = display_map.copy()
        if pos is not None and footprint is not None:
            color = (0, 0, 255) if "WARNING" in status else (0, 255, 0)

            cv2.polylines(viz_map, [footprint], True, color, 3, cv2.LINE_AA)
            cv2.circle(viz_map, pos, 8, color, -1)
            cv2.putText(viz_map, status, (pos[0] - 50, pos[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        viz_map_resized = cv2.resize(viz_map, (1024, 768))
        cv2.imshow("Drone Geofence Map", viz_map_resized)
        cv2.imshow("Live FPV Feed", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
