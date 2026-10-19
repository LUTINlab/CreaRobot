import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy
from cv_bridge import CvBridge
import cv2
import numpy as np
import mss
import os
import datetime

from . import config

class ProjectionNode(Node):
    def __init__(self):
        super().__init__('projection_node')
        self.bridge = CvBridge()

        # --- CONFIGURATION ---
        self.window_name = "Projection_QT"
        self.current_image = None
        self._generated_dir = None
        self._image_index = 0
        self.clear_timer = None

        # --- DÉPLOIEMENT FENÊTRE ---
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        projector = self.find_projector()
        if projector is not None:
            self.get_logger().info(
                f"Projecteur détecté ({projector['width']}x{projector['height']} "
                f"en {projector['left']},{projector['top']}). Déploiement plein écran..."
            )
            cv2.moveWindow(self.window_name, projector['left'], projector['top'])
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            self.get_logger().warning("Projecteur NON DÉTECTÉ — fenêtre sur écran principal.")

        # Subscriber image
        self.subscription = self.create_subscription(Image, '/pc/projector/image', self.image_callback, 10)

        # Chemin de session (latché) publié par brain_node
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/pc/session_dir', self._on_session_dir, _latched_qos)

        # Timer pour rafraîchir l'image (20 FPS)
        self.gui_timer = self.create_timer(0.05, self.update_gui)

        self.get_logger().info("Projection initialisée")

    def _on_session_dir(self, msg):
        if self._generated_dir is not None:
            return
        self._generated_dir = os.path.join(msg.data, "dessins_generes")
        os.makedirs(self._generated_dir, exist_ok=True)
        self.get_logger().info(f"Dossier dessins générés : {self._generated_dir}")

    def find_projector(self):
        """ Renvoie la géométrie du second écran (projecteur), ou None s'il n'y en a pas.
        mss lit les écrans sur Linux (X11) comme sur macOS ; l'écran principal est en (0, 0). """
        try:
            with (getattr(mss, "MSS", None) or mss.mss)() as sct:
                screens = sct.monitors[1:]  # monitors[0] = bureau virtuel entier
        except Exception as e:
            self.get_logger().warning(f"Lecture des écrans impossible : {e}")
            return None
        if len(screens) < 2:
            return None
        others = [m for m in screens if (m['left'], m['top']) != (0, 0)]
        return others[0] if others else screens[1]

    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self._generated_dir is not None:
                self._image_index += 1
                ts = datetime.datetime.now().strftime("%H-%M-%S")
                fname = f"{self._image_index:02d}_genere_{ts}.jpg"
                cv2.imwrite(os.path.join(self._generated_dir, fname), self.current_image)
                self.get_logger().info(f"Dessin généré sauvegardé : {fname}")

            # Efface la projection après PROJECTION_DISPLAY_DURATION s, indépendamment
            # de la transition de phase (gérée par ailleurs via son propre timeout).
            if self.clear_timer:
                self.clear_timer.cancel()
            self.clear_timer = self.create_timer(config.PROJECTION_DISPLAY_DURATION, self._clear_projection)
        except Exception as e:
            self.get_logger().error(f"Erreur décodage : {e}")

    def _clear_projection(self):
        if self.clear_timer:
            self.clear_timer.cancel()
            self.clear_timer = None
        self.current_image = None

    def update_gui(self):
        if self.current_image is not None:
            cv2.imshow(self.window_name, self.current_image)
        else:
            cv2.imshow(self.window_name, np.full((1080, 1920, 3), 255, dtype=np.uint8))
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()