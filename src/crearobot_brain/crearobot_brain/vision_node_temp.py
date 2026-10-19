import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy
import cv2
import base64
import numpy as np
import mss # bibliothèque de capture d'écran
import os
import datetime

# mss >= 10 expose mss.MSS (mss.mss est déprécié), les versions antérieures seulement mss.mss
_MSS = getattr(mss, "MSS", None) or mss.mss

class VisionScreenNode(Node):
    def __init__(self):
        super().__init__('vision_screen_node')

        self._photo_index = 0
        self._photos_dir  = None  # fixé quand on reçoit /pc/session_dir

        # Publisher pour le Brain
        self.image_b64_pub = self.create_publisher(String, '/pc/vision/image_raw_b64', 10)
        
        # Écoute du trigger
        self.trigger_sub = self.create_subscription(String, '/pc/camera/trigger', self.handle_trigger, 10)

        # S'abonne au chemin de session publié par brain_node (latched)
        _latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/pc/session_dir', self._on_session_dir, _latched_qos)

        self.get_logger().info("Vision Screen Node prêt (Mode HDMI)")

    def _on_session_dir(self, msg):
        if self._photos_dir is not None:
            return  # déjà reçu
        self._photos_dir = os.path.join(msg.data, "dessins")
        os.makedirs(self._photos_dir, exist_ok=True)
        self.get_logger().info(f"Dossier captures : {self._photos_dir}")

    def handle_trigger(self, msg):
        if msg.data == "ANALYZE_IMAGE":
            self.get_logger().info("Trigger reçu, capture de l'écran HDMI...")
            
            try:
                with _MSS() as sct:
                    # Mettre le 2eme ecran en HMDI sur Ecran 2
                    monitor_number = 2 
                    
                    if monitor_number >= len(sct.monitors):
                        self.get_logger().error(f"Écran {monitor_number} introuvable.")
                        return

                    monitor = sct.monitors[monitor_number]
                    
                    # Capture de l'ecran
                    screenshot = sct.grab(monitor)
                    
                    # Conversion en format OpenCV (BGR)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Sauvegarde horodatée dans le dossier de session
                    if self._photos_dir is not None:
                        self._photo_index += 1
                        ts    = datetime.datetime.now().strftime("%H-%M-%S")
                        fname = f"{self._photo_index:02d}_ecran_{ts}.jpg"
                        cv2.imwrite(os.path.join(self._photos_dir, fname), frame)
                        self.get_logger().info(f"Capture sauvegardée : {fname}")
                    else:
                        self.get_logger().warn("Dossier de session pas encore reçu, capture non sauvegardée.")
                    
                    self.send_image_to_brain(frame)
                    
            except Exception as e:
                self.get_logger().error(f"Échec de la capture d'écran : {e}")

    def send_image_to_brain(self, frame):
        try:
            # Encodage JPG
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Passage en Base64
            base64_str = base64.b64encode(buffer).decode('utf-8')
            
            msg = String()
            msg.data = base64_str
            self.image_b64_pub.publish(msg)
            
            self.get_logger().info("Capture d'écran envoyée au Brain")
            
        except Exception as e:
            self.get_logger().error(f"Erreur encodage : {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionScreenNode() 

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        
        os._exit(0)